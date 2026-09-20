// Module GPU pour CUDA acceleration de secp256k1

use rayon::prelude::*;
use rustacuda::device::DeviceAttribute;
use rustacuda::launch;
use rustacuda::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use rustacuda::prelude::*;
use secp256k1::{PublicKey, Scalar, Secp256k1, SecretKey};
use std::error::Error;
use std::ffi::CString;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

/// Largeur par defaut des fenetres de la table a base fixe (digits signes).
/// W bits -> (256/W + 1) fenetres x 2^(W-1) points x 64 octets, et 256/W + 1 additions par cle :
///   16 -> 17 additions,  36 Mo      20 -> 13 additions, 436 Mo
///   18 -> 15 additions, 126 Mo      22 -> 12 additions, 1.6 Go
/// Modifiable a l'execution avec la variable d'environnement GPU_WINDOW_BITS.
/// La table est mise en cache sur disque (target/gpu_tables/) : la construire prend
/// plusieurs secondes pour les grandes fenetres.
const DEFAULT_WINDOW_BITS: u32 = 22;

fn window_bits_from_env() -> u32 {
    let bits = match std::env::var("GPU_WINDOW_BITS") {
        Ok(v) => v.parse().expect("GPU_WINDOW_BITS must be a number"),
        Err(_) => DEFAULT_WINDOW_BITS,
    };
    assert!((4..=24).contains(&bits), "GPU_WINDOW_BITS must be in 4..=24");
    bits
}

/// Threads par bloc CUDA (modifiable avec GPU_BLOCK_SIZE pour les essais)
fn block_size() -> u32 {
    match std::env::var("GPU_BLOCK_SIZE") {
        Ok(v) => v.parse().expect("GPU_BLOCK_SIZE must be a number"),
        Err(_) => 128,
    }
}

/// Nombre de lots HASH160 pouvant etre en vol simultanement (voir `launch_hash160`)
pub const NUM_SLOTS: usize = 2;

/// Cles traitees par thread GPU : valeur compilee dans le kernel, transmise par build.rs
fn keys_per_thread() -> u32 {
    env!("GPU_KEYS_PER_THREAD").parse().expect("set by build.rs")
}

/// Un lot asynchrone : son propre stream, de la memoire hote verrouillee (copies
/// asynchrones, sans passer par un tampon intermediaire du driver) et ses buffers GPU.
/// La sortie est generique : HASH160 (20 ou 6 x 20 octets par cle) ou drapeaux du filtre.
struct Slot {
    stream: Stream,
    h_keys: LockedBuffer<u8>,
    d_keys: DeviceBuffer<u8>,
    capacity: usize,
    h_out: LockedBuffer<u8>,
    d_out: DeviceBuffer<u8>,
    /// (nombre de cles, octets de sortie) du lot lance et pas encore recupere
    in_flight: Option<(usize, usize)>,
}

impl Slot {
    fn with_capacity(num_keys: usize) -> Result<Self, Box<dyn Error>> {
        unsafe {
            Ok(Slot {
                stream: Stream::new(StreamFlags::NON_BLOCKING, None)?,
                h_keys: LockedBuffer::uninitialized(num_keys * 32)?,
                d_keys: DeviceBuffer::uninitialized(num_keys * 32)?,
                capacity: num_keys,
                h_out: LockedBuffer::uninitialized(1)?,
                d_out: DeviceBuffer::uninitialized(1)?,
                in_flight: None,
            })
        }
    }

    /// Agrandit les buffers de sortie si besoin (jamais pendant un lot en vol)
    fn ensure_out(&mut self, bytes: usize) -> Result<(), Box<dyn Error>> {
        if self.h_out.len() < bytes {
            unsafe {
                self.h_out = LockedBuffer::uninitialized(bytes)?;
                self.d_out = DeviceBuffer::uninitialized(bytes)?;
            }
        }
        Ok(())
    }
}

/// Ce qu'un lot calcule sur le GPU
#[derive(Clone, Copy)]
enum SlotJob {
    Hash160,
    Match,
}

/// Intervalle de HASH160 [lo, hi], bornes incluses, compare comme un entier big-endian
pub type Hash160Range = ([u8; 20], [u8; 20]);

/// Buffers GPU reutilises d'un appel a l'autre (cudaMalloc/cudaFree sont lents)
struct BatchBuffers {
    private_keys: DeviceBuffer<u8>,
    public_x: DeviceBuffer<u8>,
    public_y: DeviceBuffer<u8>,
    capacity: usize,
}

impl BatchBuffers {
    fn with_capacity(num_keys: usize) -> Result<Self, Box<dyn Error>> {
        unsafe {
            Ok(BatchBuffers {
                private_keys: DeviceBuffer::uninitialized(num_keys * 32)?,
                public_x: DeviceBuffer::uninitialized(num_keys * 32)?,
                public_y: DeviceBuffer::uninitialized(num_keys * 32)?,
                capacity: num_keys,
            })
        }
    }
}

pub struct GpuContext {
    slots: [Option<Slot>; NUM_SLOTS],
    buffers: Option<BatchBuffers>,
    table: DeviceBuffer<u32>,
    window_bits: u32,
    /// Intervalles du filtre GPU (10 mots par intervalle) et leur nombre
    match_ranges: Option<(DeviceBuffer<u32>, u32)>,
    module: Module,
    stream: Stream,
    _context: Context,
}

/// Convertit 32 octets big-endian en 8 limbs u32 little-endian (format du kernel)
fn be_bytes_to_limbs(bytes: &[u8], out: &mut [u32]) {
    for i in 0..8 {
        let b = &bytes[28 - 4 * i..32 - 4 * i];
        out[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
    }
}

/// Construit la table d * 2^(W*w) * G (affine, 16 limbs par point) pour chaque fenetre w
/// et chaque digit d dans 1..=2^(W-1). Les digits negatifs utilisent -P cote GPU.
fn build_base_table(window_bits: u32) -> Vec<u32> {
    let secp = Secp256k1::new();
    let num_windows = (256 / window_bits + 1) as usize;
    let entries = 1usize << (window_bits - 1);

    // bases[w] = 2^(W*w) * G, par multiplications successives par 2^W (mod n)
    let mut shift = [0u8; 32];
    shift[31 - window_bits as usize / 8] = 1 << (window_bits % 8);
    let shift = Scalar::from_be_bytes(shift).expect("2^W < n");
    let mut one = [0u8; 32];
    one[31] = 1;
    let mut bases = vec![PublicKey::from_secret_key(&secp, &SecretKey::from_slice(&one).unwrap())];
    for w in 1..num_windows {
        bases.push(bases[w - 1].mul_tweak(&secp, &shift).expect("2^(W*w) * G != infinity"));
    }

    let mut table = vec![0u32; num_windows * entries * 16];

    // Morceaux alignes sur les fenetres (tailles en puissances de 2), remplis en parallele
    let chunk_entries = entries.min(4096);
    table
        .par_chunks_mut(chunk_entries * 16)
        .enumerate()
        .for_each(|(c, chunk)| {
            let w = c * chunk_entries / entries;
            let first_digit = c * chunk_entries % entries + 1;

            // Plus grand digit possible dans cette fenetre (retenue comprise) : la derniere
            // fenetre n'a que quelques bits utiles, le reste de sa table n'est jamais lu.
            let avail_bits = 256usize.saturating_sub(window_bits as usize * w).min(window_bits as usize);
            let max_digit = entries.min(1 << avail_bits);
            if first_digit > max_digit {
                return;
            }

            // current = first_digit * base, puis + base a chaque entree
            let mut digit = [0u8; 32];
            digit[24..].copy_from_slice(&(first_digit as u64).to_be_bytes());
            let digit = Scalar::from_be_bytes(digit).expect("digit < n");
            let mut current = bases[w].mul_tweak(&secp, &digit).expect("d * base != infinity");

            for (i, limbs) in chunk.chunks_mut(16).enumerate() {
                if first_digit + i > max_digit {
                    break;
                }
                if i > 0 {
                    current = current.combine(&bases[w]).expect("d * base != infinity");
                }
                let ser = current.serialize_uncompressed();
                be_bytes_to_limbs(&ser[1..33], &mut limbs[..8]);
                be_bytes_to_limbs(&ser[33..65], &mut limbs[8..]);
            }
        });

    table
}

fn as_bytes_mut(table: &mut [u32]) -> &mut [u8] {
    // u32 -> u8 : alignement plus faible, meme zone memoire
    unsafe { std::slice::from_raw_parts_mut(table.as_mut_ptr() as *mut u8, table.len() * 4) }
}

/// Charge la table depuis le cache disque, ou la construit et l'y enregistre.
/// Le fichier contient les limbs u32 little-endian tels quels.
fn load_or_build_base_table(window_bits: u32) -> Vec<u32> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("gpu_tables");
    let path = dir.join(format!("secp256k1_base_w{}.bin", window_bits));

    let num_windows = (256 / window_bits + 1) as usize;
    let len = num_windows * (1usize << (window_bits - 1)) * 16;

    if let Ok(mut file) = File::open(&path) {
        if file.metadata().map(|m| m.len()).ok() == Some(len as u64 * 4) {
            let mut table = vec![0u32; len];
            if file.read_exact(as_bytes_mut(&mut table)).is_ok() && table_looks_valid(&table) {
                return table;
            }
        }
        println!("[GPU] Ignoring invalid table cache {}", path.display());
    }

    let mut table = build_base_table(window_bits);
    let saved = fs::create_dir_all(&dir).and_then(|_| fs::write(&path, as_bytes_mut(&mut table)));
    if let Err(e) = saved {
        println!("[GPU] Could not cache table to {}: {}", path.display(), e);
    }
    table
}

/// Controle rapide du cache : la premiere entree doit etre G
fn table_looks_valid(table: &[u32]) -> bool {
    const GX: [u32; 8] = [0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
                          0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E];
    const GY: [u32; 8] = [0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
                          0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77];
    table[..8] == GX && table[8..16] == GY
}

/// Choisit, parmi les PTX compiles par build.rs (un par architecture), le plus proche du GPU :
/// le driver sait recompiler un PTX pour un GPU plus recent que sa cible, pas plus ancien.
///   GTX 1080 Ti = compute capability 6.1 -> sm_61,  RTX 50XX = 12.0 -> sm_120
fn select_ptx(device: &Device) -> Result<String, Box<dyn Error>> {
    let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
    let capability = (major * 10 + minor) as u32;

    let arch = env!("CUDA_KERNEL_ARCHS")
        .split(',')
        .map(|a| a.parse::<u32>().expect("set by build.rs"))
        .filter(|&a| a <= capability)
        .max()
        .ok_or_else(|| format!(
            "no CUDA kernel for compute capability {}.{} (compiled: sm_{}), rebuild with CUDA_ARCH=sm_{}",
            major, minor, env!("CUDA_KERNEL_ARCHS").replace(',', ", sm_"), capability
        ))?;

    let path = Path::new(env!("CUDA_KERNEL_DIR")).join(format!("secp256k1_kernel.sm_{}.ptx", arch));
    Ok(path.to_str().expect("UTF-8 path").to_string())
}

impl GpuContext {
    /// Initialiser le contexte GPU
    pub fn new() -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;

        // Selectionner GPU 0
        let device = Device::get_device(0)?;
        let _context = Context::create_and_push(ContextFlags::MAP_HOST, device)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;

        // Charger le module PTX
        let ptx_path = select_ptx(&device)?;
        println!("[GPU] Kernel: {}", ptx_path);
        let ptx_cstr = CString::new(ptx_path)?;
        let module = Module::load_from_file(&ptx_cstr)?;

        // Table a base fixe, envoyee une seule fois
        let window_bits = window_bits_from_env();
        let host_table = load_or_build_base_table(window_bits);
        let table = DeviceBuffer::from_slice(&host_table)?;

        println!("[GPU] Initialized: Device={:?}", device.name()?);

        Ok(GpuContext {
            slots: Default::default(),
            buffers: None,
            table,
            window_bits,
            match_ranges: None,
            module,
            stream,
            _context,
        })
    }

    /// Deriver les cles publiques en batch sur GPU.
    /// Entree : cles privees de 32 octets big-endian, valides (0 < k < n).
    /// Sortie : (X, Y), 32 octets big-endian par cle.
    pub fn derive_public_keys_batch(
        &mut self,
        private_keys: &[u8],
        num_keys: usize,
    ) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
        assert_eq!(private_keys.len(), num_keys * 32);

        let mut public_x = vec![0u8; num_keys * 32];
        let mut public_y = vec![0u8; num_keys * 32];
        if num_keys == 0 {
            return Ok((public_x, public_y));
        }

        // (Re)allouer seulement si le lot est plus grand que les precedents
        if self.buffers.as_ref().map_or(true, |b| b.capacity < num_keys) {
            self.buffers = None;
            self.buffers = Some(BatchBuffers::with_capacity(num_keys)?);
        }
        let buffers = self.buffers.as_mut().unwrap();
        let bytes = num_keys * 32;

        buffers.private_keys[..bytes].copy_from(private_keys)?;

        let num_blocks = Self::num_blocks(num_keys);
        let module = &self.module;
        let stream = &self.stream;
        unsafe {
            launch!(module.derive_public_keys_batch<<<num_blocks, block_size(), 0, stream>>>(
                buffers.private_keys.as_device_ptr(),
                self.table.as_device_ptr(),
                buffers.public_x.as_device_ptr(),
                buffers.public_y.as_device_ptr(),
                num_keys as u32,
                self.window_bits
            ))?;
        }

        self.stream.synchronize()?;

        buffers.public_x[..bytes].copy_to(&mut public_x[..])?;
        buffers.public_y[..bytes].copy_to(&mut public_y[..])?;

        Ok((public_x, public_y))
    }

    /// Chaque thread GPU traite `keys_per_thread()` cles
    fn num_blocks(num_keys: usize) -> u32 {
        let num_threads = (num_keys as u32 + keys_per_thread() - 1) / keys_per_thread();
        (num_threads + block_size() - 1) / block_size()
    }

    // ---- Lots asynchrones ----
    //
    // Usage, en alternant les slots pour que le GPU ne soit jamais a l'arret :
    //   slot_keys_mut(s, n)      -> remplir les cles directement en memoire verrouillee
    //   launch_hash160(s, n, ..) -> copie H2D + kernel + copie D2H mis en file, retour immediat
    //   (ou launch_match)
    //   wait_slot(s)             -> attend la fin du lot, donne acces aux cles et a la sortie
    //
    // Avec `endo`, chaque cle privee k donne 6 cles derivees (voir src/endo.rs) pour le
    // prix d'un seul k*G.

    /// Tampon (memoire hote verrouillee) ou ecrire les `num_keys` cles privees du slot.
    /// Cles de 32 octets big-endian, valides (0 < k < n).
    pub fn slot_keys_mut(&mut self, slot: usize, num_keys: usize) -> Result<&mut [u8], Box<dyn Error>> {
        let entry = &mut self.slots[slot];
        if let Some(s) = entry.as_ref() {
            assert!(s.in_flight.is_none(), "slot {} has a batch in flight", slot);
        }

        // (Re)allouer seulement si le lot est plus grand que les precedents
        if entry.as_ref().map_or(true, |s| s.capacity < num_keys) {
            *entry = None;
            *entry = Some(Slot::with_capacity(num_keys)?);
        }
        Ok(&mut entry.as_mut().unwrap().h_keys[..num_keys * 32])
    }

    /// Lance le HASH160 (cle publique compressee) des `num_keys` cles du slot.
    /// Sortie : 20 octets par cle, ou 6 x 20 avec `endo` (ordre de src/endo.rs).
    pub fn launch_hash160(&mut self, slot: usize, num_keys: usize, endo: bool) -> Result<(), Box<dyn Error>> {
        let out_bytes = num_keys * 20 * if endo { 6 } else { 1 };
        self.launch(slot, num_keys, endo, SlotJob::Hash160, out_bytes)
    }

    /// Definit les intervalles de HASH160 retenus par `launch_match`
    pub fn set_match_ranges(&mut self, ranges: &[Hash160Range]) -> Result<(), Box<dyn Error>> {
        assert!(!ranges.is_empty());
        let mut words = Vec::with_capacity(ranges.len() * 10);
        for (lo, hi) in ranges {
            for bound in [lo, hi] {
                words.extend(bound.chunks(4).map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]])));
            }
        }
        self.match_ranges = Some((DeviceBuffer::from_slice(&words)?, ranges.len() as u32));
        Ok(())
    }

    /// Lance le filtre : le GPU calcule les HASH160 (x6 avec `endo`) et ne renvoie qu'UN
    /// octet par cle privee, bit i = la cle derivee i tombe dans un intervalle.
    pub fn launch_match(&mut self, slot: usize, num_keys: usize, endo: bool) -> Result<(), Box<dyn Error>> {
        assert!(self.match_ranges.is_some(), "call set_match_ranges first");
        self.launch(slot, num_keys, endo, SlotJob::Match, num_keys)
    }

    /// Tout est mis en file sur le stream du slot : l'appel retourne sans attendre.
    fn launch(
        &mut self,
        slot: usize,
        num_keys: usize,
        endo: bool,
        job: SlotJob,
        out_bytes: usize,
    ) -> Result<(), Box<dyn Error>> {
        let s = self.slots[slot].as_mut().expect("call slot_keys_mut first");
        assert!(s.in_flight.is_none(), "slot {} has a batch in flight", slot);
        assert!(num_keys > 0 && num_keys <= s.capacity);
        s.ensure_out(out_bytes)?;

        let num_blocks = Self::num_blocks(num_keys);
        let module = &self.module;
        let stream = &s.stream;
        // Les buffers appartiennent au slot et restent en place jusqu'a wait_slot
        unsafe {
            s.d_keys[..num_keys * 32].async_copy_from(&s.h_keys[..num_keys * 32], stream)?;

            match job {
                SlotJob::Hash160 => launch!(module.derive_hash160_batch<<<num_blocks, block_size(), 0, stream>>>(
                    s.d_keys.as_device_ptr(),
                    self.table.as_device_ptr(),
                    s.d_out.as_device_ptr(),
                    num_keys as u32,
                    self.window_bits,
                    endo as u32
                ))?,
                SlotJob::Match => {
                    let (ranges, num_ranges) = self.match_ranges.as_mut().unwrap();
                    launch!(module.derive_match_batch<<<num_blocks, block_size(), 0, stream>>>(
                        s.d_keys.as_device_ptr(),
                        self.table.as_device_ptr(),
                        ranges.as_device_ptr(),
                        *num_ranges,
                        s.d_out.as_device_ptr(),
                        num_keys as u32,
                        self.window_bits,
                        endo as u32
                    ))?
                }
            }

            s.d_out[..out_bytes].async_copy_to(&mut s.h_out[..out_bytes], stream)?;
        }

        s.in_flight = Some((num_keys, out_bytes));
        Ok(())
    }

    /// Attend la fin du lot du slot. Retourne (cles privees, sortie du lot).
    /// Une cle nulle (ou multiple de n) donne des HASH160 a zero / aucun drapeau.
    pub fn wait_slot(&mut self, slot: usize) -> Result<(&[u8], &[u8]), Box<dyn Error>> {
        let s = self.slots[slot].as_mut().expect("call slot_keys_mut first");
        let (num_keys, out_bytes) = s.in_flight.take().expect("no batch in flight");

        s.stream.synchronize()?;

        Ok((&s.h_keys[..num_keys * 32], &s.h_out[..out_bytes]))
    }

    /// Version synchrone simple : HASH160 de chaque cle, 20 octets par cle
    /// (6 x 20 octets par cle avec `endo`)
    pub fn derive_hash160_batch(
        &mut self,
        private_keys: &[u8],
        num_keys: usize,
        endo: bool,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        assert_eq!(private_keys.len(), num_keys * 32);
        if num_keys == 0 {
            return Ok(Vec::new());
        }

        self.slot_keys_mut(0, num_keys)?.copy_from_slice(private_keys);
        self.launch_hash160(0, num_keys, endo)?;
        let (_, hashes) = self.wait_slot(0)?;
        Ok(hashes.to_vec())
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        // Ne pas liberer la memoire d'un lot encore en vol
        for slot in self.slots.iter().flatten() {
            let _ = slot.stream.synchronize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_init() {
        match GpuContext::new() {
            Ok(_ctx) => println!("GPU context created successfully"),
            Err(e) => println!("GPU not available: {}", e),
        }
    }
}
