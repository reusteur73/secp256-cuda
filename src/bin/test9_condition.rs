// Benchmark de la chaîne complète avec une CONDITION testée sur chaque résultat :
//   [thread RNG]  clés aléatoires préparées en avance (pool dédié, hors du chemin critique)
//   [thread GPU]  clés -> mémoire verrouillée -> k*G + HASH160 sur GPU (2 lots en vol)
//   [main/Rayon]  condition -> pour les résultats retenus : adresse + clé
//
// Trois modes, pour comparer :
//   cpu    : le GPU renvoie 20 octets par clé, la condition est testée sur CPU (Rayon)
//   single : le GPU pré-filtre lui-même et ne renvoie qu'un octet de drapeaux par clé ;
//            le CPU ne recalcule et ne vérifie que les candidats signalés
//   endo   : comme single, mais chaque clé aléatoire k donne 6 clés dérivées
//            (+-k, +-lambda*k, +-lambda^2*k, voir src/endo.rs) pour un seul k*G
//
// Usage: cargo run --bin test9_condition --release -- [num_keys] [batch_size] [prefix] [mode]
//   ex:  cargo run --bin test9_condition --release -- 64000000 1000000 1GPU endo

use secp256k1::{Secp256k1, SecretKey, PublicKey};
use sha2::{Sha256, Digest};
use ripemd::Ripemd160;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use utxo_loader::address::{p2pkh_address, prefix_hash160_ranges};
use utxo_loader::endo::{derived_private_key, NUM_DERIVED};
use utxo_loader::gpu::Hash160Range;

mod gpu {
    pub use utxo_loader::gpu::*;
}

/// Condition testée sur chaque clé générée, à partir du HASH160 de sa clé publique.
/// Appelée en parallèle (Rayon) : doit être rapide et sans état mutable.
trait Condition: Sync {
    fn describe(&self) -> String;

    /// Test exact
    fn matches(&self, hash160: &[u8; 20]) -> bool;

    /// Pré-filtre pour le GPU : intervalles de HASH160 contenant tous les résultats retenus
    /// par `matches` (il peut être un peu plus large, jamais plus étroit)
    fn gpu_ranges(&self) -> Vec<Hash160Range>;
}

/// Adresse « vanity » : l'adresse P2PKH commence par un préfixe choisi
struct AddressPrefix {
    prefix: String,
    ranges: Vec<Hash160Range>,
}

impl AddressPrefix {
    fn new(prefix: &str) -> Result<Self, String> {
        let ranges = prefix_hash160_ranges(prefix)?;
        Ok(AddressPrefix { prefix: prefix.to_string(), ranges })
    }
}

impl Condition for AddressPrefix {
    fn describe(&self) -> String {
        format!("address starts with \"{}\"", self.prefix)
    }

    fn matches(&self, hash160: &[u8; 20]) -> bool {
        p2pkh_address(hash160).as_str().starts_with(&self.prefix)
    }

    fn gpu_ranges(&self) -> Vec<Hash160Range> {
        self.ranges.clone()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Cpu,
    Single,
    Endo,
}

impl Mode {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "cpu" => Ok(Mode::Cpu),
            "single" => Ok(Mode::Single),
            "endo" => Ok(Mode::Endo),
            _ => Err(format!("unknown mode \"{}\" (cpu, single, endo)", s)),
        }
    }

    /// Clés testées par clé aléatoire tirée
    fn keys_per_draw(self) -> usize {
        if self == Mode::Endo { NUM_DERIVED } else { 1 }
    }
}

/// Clé privée au format WIF (clé publique compressée)
fn wif(private_key: &[u8]) -> String {
    let mut full = vec![0x80];
    full.extend(private_key);
    full.push(0x01);
    let checksum = Sha256::digest(Sha256::digest(&full));
    full.extend(&checksum[..4]);
    bs58::encode(full).into_string()
}

/// Référence : HASH160 entièrement sur CPU avec libsecp256k1
fn cpu_reference_hash160(private_key: &SecretKey) -> [u8; 20] {
    let secp = Secp256k1::new();
    let pubkey_bytes = PublicKey::from_secret_key(&secp, private_key).serialize();
    Ripemd160::digest(Sha256::digest(pubkey_bytes)).into()
}

struct GpuBatch {
    private_keys: Vec<u8>,
    /// HASH160 (mode cpu) ou un octet de drapeaux par clé (modes single / endo)
    output: Vec<u8>,
}

/// Remplit `keys` d'octets aléatoires en parallèle. StdRng (ChaCha) est un générateur
/// cryptographique, initialisé depuis l'OS pour chaque morceau : bien plus rapide
/// qu'un appel système OsRng pour tout le tampon.
///
/// Tourne dans un pool dédié : dans le pool global, le tirage ferait la queue derrière
/// le calcul de la condition et le GPU attendrait ses clés.
fn fill_random(pool: &rayon::ThreadPool, keys: &mut [u8]) {
    pool.install(|| {
        keys.par_chunks_mut(1 << 20)
            .for_each_init(StdRng::from_entropy, |rng, chunk| rng.fill_bytes(chunk));
    });
}

const RNG_THREADS: usize = 4;

#[derive(Default)]
struct GpuThreadStats {
    starved: Duration,
    upload: Duration,
    wait: Duration,
    copy: Duration,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let num_keys: usize = args.next().map(|a| a.parse().expect("num_keys")).unwrap_or(16_000_000);
    let batch_size: usize = args.next().map(|a| a.parse().expect("batch_size")).unwrap_or(1_000_000);
    let condition = AddressPrefix::new(&args.next().unwrap_or_else(|| "1GPU".to_string()))?;
    let mode = Mode::parse(&args.next().unwrap_or_else(|| "endo".to_string()))?;

    println!("=== Pipeline + condition: {} random keys, batches of {} ===", num_keys, batch_size);
    println!("Condition: {}", condition.describe());
    println!("Mode: {}\n", match mode {
        Mode::Cpu => "cpu (condition on CPU, 1 key per random draw)",
        Mode::Single => "single (GPU pre-filter, 1 key per random draw)",
        Mode::Endo => "endo (GPU pre-filter, 6 derived keys per random draw)",
    });

    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let (batch_tx, batch_rx) = mpsc::sync_channel::<GpuBatch>(1);
    let (keys_tx, keys_rx) = mpsc::sync_channel::<Vec<u8>>(2);
    // Les tampons des lots traités retournent à leur producteur : pas de réallocation par lot
    let (keys_recycle_tx, keys_recycle_rx) = mpsc::channel::<Vec<u8>>();
    let (output_recycle_tx, output_recycle_rx) = mpsc::channel::<Vec<u8>>();

    let gpu_ranges = condition.gpu_ranges();
    let init_start = Instant::now();
    let gpu_thread = thread::spawn(move || -> Result<GpuThreadStats, String> {
        // Le contexte CUDA est lié au thread qui le crée : tout le GPU vit ici
        let launch = move |ctx: &mut gpu::GpuContext, slot: usize, n: usize| match mode {
            Mode::Cpu => ctx.launch_hash160(slot, n, false),
            Mode::Single => ctx.launch_match(slot, n, false),
            Mode::Endo => ctx.launch_match(slot, n, true),
        };

        let init = || -> Result<gpu::GpuContext, Box<dyn std::error::Error>> {
            let mut ctx = gpu::GpuContext::new()?;
            ctx.set_match_ranges(&gpu_ranges)?;
            // Warm-up : initialisation du kernel + allocation des slots à la taille d'un lot
            let n = batch_size.min(num_keys);
            for slot in 0..gpu::NUM_SLOTS {
                ctx.slot_keys_mut(slot, n)?.fill(1);
                launch(&mut ctx, slot, n)?;
                ctx.wait_slot(slot)?;
            }
            Ok(ctx)
        };
        let mut gpu_ctx = match init() {
            Ok(ctx) => ctx,
            Err(e) => {
                let _ = ready_tx.send(Err(e.to_string()));
                return Err(e.to_string());
            }
        };
        let _ = ready_tx.send(Ok(()));

        let mut run = || -> Result<GpuThreadStats, Box<dyn std::error::Error>> {
            let mut stats = GpuThreadStats::default();
            let mut pending: Option<usize> = None;
            // Clés du lot en vol de chaque slot : elles suivent le lot jusqu'au consommateur
            let mut in_flight: [Option<Vec<u8>>; gpu::NUM_SLOTS] = Default::default();
            let mut slot = 0;

            loop {
                // Lancer le lot suivant AVANT d'attendre le précédent : le GPU enchaîne
                // sans temps mort.
                let mut launched = None;
                let t = Instant::now();
                let next_keys = keys_rx.recv().ok();
                stats.starved += t.elapsed();

                if let Some(keys) = next_keys {
                    let n = keys.len() / 32;

                    let t = Instant::now();
                    gpu_ctx.slot_keys_mut(slot, n)?.copy_from_slice(&keys);
                    stats.upload += t.elapsed();

                    launch(&mut gpu_ctx, slot, n)?;
                    in_flight[slot] = Some(keys);
                    launched = Some(slot);
                }

                if let Some(p) = pending {
                    let t = Instant::now();
                    let (_, gpu_output) = gpu_ctx.wait_slot(p)?;
                    stats.wait += t.elapsed();

                    let t = Instant::now();
                    let mut output = output_recycle_rx.try_recv().unwrap_or_default();
                    output.clear();
                    output.extend_from_slice(gpu_output);
                    stats.copy += t.elapsed();

                    let private_keys = in_flight[p].take().expect("keys of the in-flight batch");
                    if batch_tx.send(GpuBatch { private_keys, output }).is_err() {
                        break;
                    }
                }

                pending = launched;
                if pending.is_none() {
                    break;
                }
                slot = (slot + 1) % gpu::NUM_SLOTS;
            }
            Ok(stats)
        };
        run().map_err(|e| e.to_string())
    });

    ready_rx.recv()??;
    println!("[GPU] Init + warm-up: {:.0} ms\n", init_start.elapsed().as_secs_f64() * 1000.0);

    let total_start = Instant::now();

    let rng_thread = thread::spawn(move || -> Duration {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(RNG_THREADS).build().expect("rng pool");
        let mut busy = Duration::ZERO;
        let mut done = 0;
        while done < num_keys {
            let n = batch_size.min(num_keys - done);

            let t = Instant::now();
            let mut keys = keys_recycle_rx.try_recv().unwrap_or_default();
            keys.resize(n * 32, 0);
            fill_random(&pool, &mut keys);
            busy += t.elapsed();

            if keys_tx.send(keys).is_err() {
                break;
            }
            done += n;
        }
        busy
    });

    let mut t_condition = Duration::ZERO;
    let mut generated = 0usize;
    let mut num_candidates = 0usize;
    let mut num_matches = 0;

    for batch in batch_rx {
        let n = batch.private_keys.len() / 32;
        let key_at = |i: usize| SecretKey::from_slice(&batch.private_keys[i * 32..(i + 1) * 32]);

        // Résultats à examiner : (numéro de clé, indice de clé dérivée)
        let t = Instant::now();
        let candidates: Vec<(usize, usize)> = if mode == Mode::Cpu {
            // Condition exacte sur chaque HASH160 (CPU Rayon)
            let hashes: &[[u8; 20]] = batch.output.as_chunks::<20>().0;
            hashes
                .par_iter()
                .enumerate()
                .filter(|(_, hash)| condition.matches(hash))
                .map(|(i, _)| (i, 0))
                .collect()
        } else {
            // Le GPU a déjà filtré : un octet par clé, presque toujours nul
            batch.output
                .iter()
                .enumerate()
                .filter(|(_, &flags)| flags != 0)
                .flat_map(|(i, &flags)| (0..NUM_DERIVED).filter(move |d| flags >> d & 1 != 0).map(move |d| (i, d)))
                .collect()
        };
        t_condition += t.elapsed();
        num_candidates += candidates.len();

        for (i, derived) in candidates {
            // Chaque candidat est recalculé sur CPU (libsecp256k1) puis testé exactement :
            // rien n'est affiché sur la seule foi du GPU
            let Ok(random_key) = key_at(i) else { continue };
            let private_key = derived_private_key(&random_key, derived);
            let hash160 = cpu_reference_hash160(&private_key);
            if mode == Mode::Cpu {
                assert_eq!(hash160, batch.output[i * 20..(i + 1) * 20], "HASH160 mismatch on match");
            }
            if !condition.matches(&hash160) {
                continue; // HASH160 sur une borne du pré-filtre
            }

            num_matches += 1;
            if num_matches <= 5 {
                let key_bytes = private_key.secret_bytes();
                println!("MATCH  {}", p2pkh_address(&hash160).as_str());
                println!("       key: {}  (WIF: {})", hex::encode(key_bytes), wif(&key_bytes));
            }
        }

        generated += n;
        let _ = keys_recycle_tx.send(batch.private_keys);
        let _ = output_recycle_tx.send(batch.output);
    }
    let total = total_start.elapsed();

    let t_random = rng_thread.join().expect("rng thread panicked");
    let stats = gpu_thread.join().expect("gpu thread panicked")?;
    assert_eq!(generated, num_keys);

    if num_matches > 5 {
        println!("... and {} more", num_matches - 5);
    }
    let keys_tested = num_keys * mode.keys_per_draw();
    println!("\nMatches: {} / {} keys tested ({} candidates re-checked on CPU)", num_matches, keys_tested, num_candidates);
    // Doit être le même dans tous les modes : une clé dérivée a la même chance qu'une clé tirée
    println!("Observed rate: 1 match per {:.0} keys tested\n", keys_tested as f64 / num_matches.max(1) as f64);

    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    println!("{:<34} {:<8} {:>9}", "Étape", "Où?", "Temps");
    println!("{:-<53}", "");
    println!("{:<34} {:<8} {:>7.0}ms", "Random bytes (en avance)", "CPU", ms(t_random));
    println!("{:<34} {:<8} {:>7.0}ms", "Thread GPU : attente des clés", "-", ms(stats.starved));
    println!("{:<34} {:<8} {:>7.0}ms", "Thread GPU : copies mémoire", "CPU", ms(stats.upload + stats.copy));
    println!("{:<34} {:<8} {:>7.0}ms", "Thread GPU : attente du GPU", "GPU", ms(stats.wait));
    println!("{:<34} {:<8} {:>7.0}ms", "Condition / lecture des drapeaux", "CPU", ms(t_condition));
    println!("{:-<53}", "");
    println!("Temps réel: {:.0} ms", ms(total));
    println!("Random keys:  {:.0} /sec", num_keys as f64 / total.as_secs_f64());
    println!("THROUGHPUT:   {:.0} keys tested/sec", keys_tested as f64 / total.as_secs_f64());

    Ok(())
}
