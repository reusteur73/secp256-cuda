use secp256k1::{Secp256k1, SecretKey, PublicKey};
use rand::rngs::OsRng;
use rand::RngCore;
use rayon::prelude::*;
use sha2::{Sha256, Digest};
use ripemd::Ripemd160;

use utxo_loader::address::prefix_hash160_ranges;
use utxo_loader::endo::{derived_private_key, NUM_DERIVED};

mod gpu {
    pub use utxo_loader::gpu::*;
}

/// Clés limites : petites valeurs, n-1, fenêtres à zéro, limbs saturés
fn edge_case_keys() -> Vec<[u8; 32]> {
    let mut keys = Vec::new();

    for k in [1u8, 2, 3, 0xFF] {
        let mut key = [0u8; 32];
        key[31] = k;
        keys.push(key);
    }

    // n - 1
    let n_minus_1 = hex::decode("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140").unwrap();
    keys.push(n_minus_1.try_into().unwrap());

    // Un seul bit à 1, à différentes positions (la plupart des fenêtres à zéro)
    for bit in [8usize, 15, 16, 31, 32, 33, 127, 128, 200, 255] {
        let mut key = [0u8; 32];
        key[31 - bit / 8] = 1 << (bit % 8);
        keys.push(key);
    }

    // Motifs saturés
    let mut key = [0xFFu8; 32];
    key[0] = 0x7F;
    keys.push(key);
    let mut key = [0u8; 32];
    key[16..].fill(0xFF);
    keys.push(key);

    keys
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU vs CPU Validation Test ===\n");

    // Nombre volontairement non multiple de la taille de lot par thread GPU
    // (argument optionnel : nombre de clés aléatoires, ex. 2000003 pour une validation longue)
    let num_random: usize = std::env::args().nth(1).map(|a| a.parse().expect("num_random")).unwrap_or(10_007);
    let mut keys = edge_case_keys();
    let num_edge = keys.len();

    let mut rng = OsRng;
    while keys.len() < num_edge + num_random {
        let mut key_bytes = [0u8; 32];
        rng.fill_bytes(&mut key_bytes);

        // S'assurer que c'est valide pour secp256k1
        if SecretKey::from_slice(&key_bytes).is_ok() {
            keys.push(key_bytes);
        }
    }
    println!("[CPU] {} edge-case keys + {} random keys", num_edge, num_random);

    println!("[CPU] Computing expected public keys...");
    let secp = Secp256k1::new();
    let cpu_results: Vec<[u8; 65]> = keys
        .par_iter()
        .map(|k| {
            let secret_key = SecretKey::from_slice(k).unwrap();
            PublicKey::from_secret_key(&secp, &secret_key).serialize_uncompressed()
        })
        .collect();

    println!("[GPU] Computing public keys on GPU...");
    let mut gpu_ctx = gpu::GpuContext::new()?;
    let private_keys_bytes_all: Vec<u8> = keys.iter().flatten().copied().collect();
    let (gpu_x, gpu_y) = gpu_ctx.derive_public_keys_batch(&private_keys_bytes_all, keys.len())?;

    println!("\n=== Comparison ===\n");

    let mut matches = 0;
    let mut mismatches = 0;

    for (i, key) in keys.iter().enumerate() {
        // Skip first byte (0x04), extract x and y
        let cpu_x = &cpu_results[i][1..33];
        let cpu_y = &cpu_results[i][33..65];

        let gpu_x_slice = &gpu_x[i*32..(i+1)*32];
        let gpu_y_slice = &gpu_y[i*32..(i+1)*32];

        if cpu_x == gpu_x_slice && cpu_y == gpu_y_slice {
            matches += 1;
        } else {
            mismatches += 1;
            if mismatches <= 10 {
                println!("✗ Key #{} {}: MISMATCH", i, hex::encode(key));
                println!("  CPU X: {}", hex::encode(cpu_x));
                println!("  GPU X: {}", hex::encode(gpu_x_slice));
                println!("  CPU Y: {}", hex::encode(cpu_y));
                println!("  GPU Y: {}", hex::encode(gpu_y_slice));
            }
        }
    }

    println!("=== Results (public keys) ===");
    println!("Matches:    {}", matches);
    println!("Mismatches: {}", mismatches);

    // HASH160 calculé sur GPU vs RIPEMD160(SHA256(clé publique compressée)) sur CPU
    println!("\n[GPU] Computing HASH160 on GPU...");
    let gpu_hashes = gpu_ctx.derive_hash160_batch(&private_keys_bytes_all, keys.len(), false)?;

    let mut hash_matches = 0;
    for (i, key) in keys.iter().enumerate() {
        let mut compressed = [0u8; 33];
        compressed[0] = 0x02 | (cpu_results[i][64] & 1);
        compressed[1..].copy_from_slice(&cpu_results[i][1..33]);
        let expected = Ripemd160::digest(Sha256::digest(compressed));

        let gpu_hash = &gpu_hashes[i*20..(i+1)*20];
        if expected.as_slice() == gpu_hash {
            hash_matches += 1;
        } else {
            mismatches += 1;
            if mismatches <= 10 {
                println!("✗ Key #{} {}: HASH160 MISMATCH", i, hex::encode(key));
                println!("  CPU: {}", hex::encode(expected));
                println!("  GPU: {}", hex::encode(gpu_hash));
            }
        }
    }

    println!("=== Results (HASH160) ===");
    println!("Matches:    {}", hash_matches);
    println!("Mismatches: {}", keys.len() - hash_matches);

    // Clés dérivées : 6 HASH160 par clé sur GPU vs clés privées dérivées recalculées sur CPU
    println!("\n[GPU] Computing derived-key HASH160 (x6) on GPU...");
    let gpu_derived = gpu_ctx.derive_hash160_batch(&private_keys_bytes_all, keys.len(), true)?;

    let cpu_derived: Vec<[[u8; 20]; NUM_DERIVED]> = keys
        .par_iter()
        .map(|k| {
            let secret_key = SecretKey::from_slice(k).unwrap();
            std::array::from_fn(|index| {
                let derived = derived_private_key(&secret_key, index);
                let compressed = PublicKey::from_secret_key(&secp, &derived).serialize();
                Ripemd160::digest(Sha256::digest(compressed)).into()
            })
        })
        .collect();

    let mut derived_matches = 0;
    for (i, key) in keys.iter().enumerate() {
        for index in 0..NUM_DERIVED {
            let offset = (i * NUM_DERIVED + index) * 20;
            if cpu_derived[i][index] == gpu_derived[offset..offset + 20] {
                derived_matches += 1;
            } else {
                mismatches += 1;
                if mismatches <= 10 {
                    println!("✗ Key #{} {} derived #{}: HASH160 MISMATCH", i, hex::encode(key), index);
                }
            }
        }
    }

    println!("=== Results (derived keys, 6 per key) ===");
    println!("Matches:    {}", derived_matches);
    println!("Mismatches: {}", keys.len() * NUM_DERIVED - derived_matches);

    // Filtre GPU : les drapeaux doivent correspondre exactement aux intervalles, calculés sur CPU
    let ranges = prefix_hash160_ranges("1A")?;
    gpu_ctx.set_match_ranges(&ranges)?;
    let mut flag_matches = 0;
    let mut flagged = 0;
    for endo in [false, true] {
        gpu_ctx.slot_keys_mut(0, keys.len())?.copy_from_slice(&private_keys_bytes_all);
        gpu_ctx.launch_match(0, keys.len(), endo)?;
        let (_, flags) = gpu_ctx.wait_slot(0)?;

        for (i, key) in keys.iter().enumerate() {
            let mut expected = 0u8;
            for index in 0..if endo { NUM_DERIVED } else { 1 } {
                let hash = cpu_derived[i][index];
                if ranges.iter().any(|(lo, hi)| *lo <= hash && hash <= *hi) {
                    expected |= 1 << index;
                }
            }

            flagged += expected.count_ones();
            if flags[i] == expected {
                flag_matches += 1;
            } else {
                mismatches += 1;
                if mismatches <= 10 {
                    println!("✗ Key #{} {}: FLAGS MISMATCH (endo={}) cpu={:06b} gpu={:06b}",
                        i, hex::encode(key), endo, expected, flags[i]);
                }
            }
        }
    }

    println!("=== Results (GPU filter flags, prefix \"1A\") ===");
    println!("Matches:    {}  ({} keys flagged)", flag_matches, flagged);
    println!("Mismatches: {}", keys.len() * 2 - flag_matches);

    if mismatches > 0 {
        println!("\n⚠️  GPU kernel produces INCORRECT results!");
        std::process::exit(1);
    }

    println!("\n✓ GPU kernel matches libsecp256k1 on all keys");
    Ok(())
}
