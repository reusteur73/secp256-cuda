// Benchmark de la chaîne complète :
//   random bytes (CPU) -> k*G (GPU) -> SHA256 + RIPEMD160 (CPU Rayon) -> Base58Check (CPU Rayon)
// Les étages tournent en parallèle (le GPU calcule le lot suivant pendant que le CPU
// hache le lot courant) ; on affiche le temps d'occupation de chaque étape.
//
// Usage: cargo run --bin test8_pipeline --release -- [num_keys] [batch_size]

use secp256k1::{Secp256k1, SecretKey, PublicKey};
use sha2::{Sha256, Digest};
use ripemd::Ripemd160;
use rand::rngs::OsRng;
use rand::RngCore;
use rayon::prelude::*;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use utxo_loader::address::{p2pkh_address, Address};

mod gpu {
    pub use utxo_loader::gpu::*;
}

/// HASH160 de la clé publique compressée (02/03 selon la parité de Y, puis X)
fn hash160_compressed(x: &[u8], y: &[u8]) -> [u8; 20] {
    let mut pubkey = [0u8; 33];
    pubkey[0] = 0x02 | (y[31] & 1);
    pubkey[1..].copy_from_slice(x);
    Ripemd160::digest(Sha256::digest(pubkey)).into()
}

/// Référence : même chaîne entièrement sur CPU avec libsecp256k1 et la crate bs58
fn cpu_reference_address(private_key: &[u8]) -> String {
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(private_key).expect("valid key");
    let pubkey_bytes = PublicKey::from_secret_key(&secp, &secret_key).serialize();

    let mut full = vec![0x00];
    full.extend(Ripemd160::digest(Sha256::digest(pubkey_bytes)));
    let checksum = Sha256::digest(Sha256::digest(&full));
    full.extend(&checksum[..4]);
    bs58::encode(full).into_string()
}

struct GpuBatch {
    private_keys: Vec<u8>,
    pub_x: Vec<u8>,
    pub_y: Vec<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let num_keys: usize = args.next().map(|a| a.parse().expect("num_keys")).unwrap_or(4_000_000);
    let batch_size: usize = args.next().map(|a| a.parse().expect("batch_size")).unwrap_or(1_000_000);

    println!("=== Full pipeline benchmark: {} keys, batches of {} ===
", num_keys, batch_size);

    // Cas limite de l'encodeur : hash160 nul -> 21 octets nuls en tête
    assert_eq!(p2pkh_address(&[0u8; 20]).as_str(), "1111111111111111111114oLvT2");

    // Trois étages qui se chevauchent, reliés par des canaux bornés :
    //   [thread RNG] --clés--> [thread GPU] --clés publiques--> [main : Rayon hash + Base58]
    // Pendant que le CPU traite le lot N, le GPU calcule déjà le lot N+1.
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let (keys_tx, keys_rx) = mpsc::sync_channel::<Vec<u8>>(1);
    let (pub_tx, pub_rx) = mpsc::sync_channel::<GpuBatch>(1);

    let init_start = Instant::now();
    let warmup_keys = batch_size.min(num_keys);
    let gpu_thread = thread::spawn(move || -> Result<Duration, String> {
        // Le contexte CUDA est lié au thread qui le crée : tout le GPU vit ici
        let init = || -> Result<gpu::GpuContext, Box<dyn std::error::Error>> {
            let mut ctx = gpu::GpuContext::new()?;
            // Warm-up : paie l'initialisation du kernel et alloue les buffers à la taille d'un lot
            ctx.derive_public_keys_batch(&vec![1u8; 32 * warmup_keys], warmup_keys)?;
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

        let mut busy = Duration::ZERO;
        for private_keys in keys_rx {
            // 2. Scalar mult k*G (GPU, transferts inclus)
            let t = Instant::now();
            let (pub_x, pub_y) = gpu_ctx
                .derive_public_keys_batch(&private_keys, private_keys.len() / 32)
                .map_err(|e| e.to_string())?;
            busy += t.elapsed();

            if pub_tx.send(GpuBatch { private_keys, pub_x, pub_y }).is_err() {
                break;
            }
        }
        Ok(busy)
    });

    ready_rx.recv()??;
    println!("[GPU] Init + warm-up: {:.0} ms
", init_start.elapsed().as_secs_f64() * 1000.0);

    let total_start = Instant::now();

    let rng_thread = thread::spawn(move || -> Duration {
        let mut rng = OsRng;
        let mut busy = Duration::ZERO;
        let mut done = 0;
        while done < num_keys {
            let n = batch_size.min(num_keys - done);

            // 1. Random bytes (CPU)
            let t = Instant::now();
            let mut private_keys = vec![0u8; n * 32];
            rng.fill_bytes(&mut private_keys);
            busy += t.elapsed();

            if keys_tx.send(private_keys).is_err() {
                break;
            }
            done += n;
        }
        busy
    });

    let mut t_hash = Duration::ZERO;
    let mut t_base58 = Duration::ZERO;
    let mut verified = 0;
    let mut generated = 0;
    let mut last_address = String::new();

    for batch in pub_rx {
        // 3. SHA256 + RIPEMD160 (CPU Rayon)
        let t = Instant::now();
        let hashes: Vec<[u8; 20]> = batch.pub_x
            .par_chunks(32)
            .zip(batch.pub_y.par_chunks(32))
            .map(|(x, y)| hash160_compressed(x, y))
            .collect();
        t_hash += t.elapsed();

        // 4. Base58Check (CPU Rayon)
        let t = Instant::now();
        let addresses: Vec<Address> = hashes.par_iter().map(p2pkh_address).collect();
        t_base58 += t.elapsed();

        // Contrôle : quelques adresses du premier lot vs chaîne 100% CPU
        if generated == 0 {
            for i in 0..addresses.len().min(100) {
                let expected = cpu_reference_address(&batch.private_keys[i * 32..(i + 1) * 32]);
                assert_eq!(addresses[i].as_str(), expected, "address mismatch for key #{}", i);
                verified += 1;
            }
        }

        last_address = addresses[addresses.len() - 1].as_str().to_string();
        generated += addresses.len();
    }
    let total = total_start.elapsed();

    let t_random = rng_thread.join().expect("rng thread panicked");
    let t_gpu = gpu_thread.join().expect("gpu thread panicked")?;
    assert_eq!(generated, num_keys);

    println!("Verified {} addresses against the pure-CPU chain ✓", verified);
    println!("Sample address: {}
", last_address);

    let stages = [
        ("Random bytes", "CPU", t_random),
        ("Scalar mult (k*G)", "GPU", t_gpu),
        ("SHA256 + RIPEMD160", "CPU", t_hash),
        ("Base58Check", "CPU", t_base58),
    ];
    let sum: Duration = stages.iter().map(|s| s.2).sum();

    println!("{:<22} {:<5} {:>10}", "Étape", "Où?", "Occupé");
    println!("{:-<40}", "");
    for (name, place, d) in stages {
        println!("{:<22} {:<5} {:>8.0}ms", name, place, d.as_secs_f64() * 1000.0);
    }
    println!("{:-<40}", "");
    println!("Somme des étapes:  {:>6.0} ms (si tout était séquentiel)", sum.as_secs_f64() * 1000.0);
    println!("Temps réel:        {:>6.0} ms (étapes en parallèle)", total.as_secs_f64() * 1000.0);
    println!("THROUGHPUT: {:.0} addresses/sec", num_keys as f64 / total.as_secs_f64());

    Ok(())
}
