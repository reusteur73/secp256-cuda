use std::io::{self, Write};
use std::time::Instant;
use rand::rngs::OsRng;
use rand::RngCore;

// Import le module GPU
mod gpu {
    pub use utxo_loader::gpu::*;
}

fn benchmark_gpu(num_keys: usize) -> Result<f64, Box<dyn std::error::Error>> {
    println!("\n=== GPU BENCHMARK ===\n");

    let start = Instant::now();

    // Initialiser GPU
    println!("[GPU] Initializing context...");
    let mut gpu_ctx = gpu::GpuContext::new()?;

    // Générer clés privées aléatoires (sur CPU)
    println!("[GPU] Generating {} random private keys on CPU...", num_keys);
    let mut private_keys = vec![0u8; num_keys * 32];
    let mut rng = OsRng;
    rng.fill_bytes(&mut private_keys);

    // Warm-up : le premier lancement paie l'initialisation du kernel
    gpu_ctx.derive_public_keys_batch(&private_keys[..32 * num_keys.min(1024)], num_keys.min(1024))?;

    // Lancer GPU
    println!("[GPU] Launching kernel...");
    let gpu_start = Instant::now();
    let (_pub_x, _pub_y) = gpu_ctx.derive_public_keys_batch(&private_keys, num_keys)?;
    let gpu_duration = gpu_start.elapsed();

    let total_duration = start.elapsed();
    let throughput = num_keys as f64 / gpu_duration.as_secs_f64();

    println!("\n[GPU] Results:");
    println!("  Total time:     {:.2} ms", total_duration.as_secs_f64() * 1000.0);
    println!("  GPU time:       {:.2} ms", gpu_duration.as_secs_f64() * 1000.0);
    println!("  Throughput:     {:.2} keys/sec", throughput);

    Ok(throughput)
}

fn benchmark_cpu(num_keys: usize) -> f64 {
    println!("\n=== CPU BENCHMARK (Rayon) ===\n");

    use secp256k1::{Secp256k1, SecretKey, PublicKey};
    use sha2::{Sha256, Digest};
    use ripemd::Ripemd160;
    use bs58;
    use rand::rngs::OsRng;
    use rayon::prelude::*;

    fn sha256(data: &[u8]) -> Vec<u8> {
        Sha256::digest(data).to_vec()
    }

    fn ripemd160(data: &[u8]) -> Vec<u8> {
        Ripemd160::digest(data).to_vec()
    }

    fn base58check_encode(payload: &[u8]) -> String {
        let checksum = sha256(&sha256(payload))[0..4].to_vec();
        let mut full = payload.to_vec();
        full.extend(checksum);
        bs58::encode(full).into_string()
    }

    fn create_p2pkh_address() -> String {
        let secp = Secp256k1::new();
        let mut rng = OsRng;
        let secret_key = SecretKey::new(&mut rng);
        let public_key = PublicKey::from_secret_key(&secp, &secret_key);
        let pubkey_bytes = public_key.serialize();
        let pubkey_hash = ripemd160(&sha256(&pubkey_bytes));
        let mut payload = vec![0x00];
        payload.extend(pubkey_hash);
        base58check_encode(&payload)
    }

    let start = Instant::now();

    let _addresses: Vec<String> = (0..num_keys)
        .into_par_iter()
        .map(|_| create_p2pkh_address())
        .collect();

    let duration = start.elapsed();
    let throughput = num_keys as f64 / duration.as_secs_f64();

    println!("[CPU] Results:");
    println!("  Total time:     {:.2} ms", duration.as_secs_f64() * 1000.0);
    println!("  Throughput:     {:.2} keys/sec", throughput);

    throughput
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Nombre de clés : argument CLI (cargo run --bin test4_gpu --release -- 1000000) ou saisie
    let num_keys: usize = match std::env::args().nth(1) {
        Some(arg) => arg.parse().expect("Please enter a valid number"),
        None => {
            let mut input = String::new();
            println!("Enter number of keys to benchmark (1000-100000): ");
            io::stdout().flush()?;
            io::stdin().read_line(&mut input)?;
            input.trim().parse().unwrap_or(10000)
        }
    };

    println!("\n{}", "=".repeat(60));
    println!("Bitcoin Wallet Generation Benchmark: GPU vs CPU");
    println!("{}\n", "=".repeat(60));

    // Test GPU
    let gpu_throughput = match benchmark_gpu(num_keys) {
        Ok(tp) => tp,
        Err(e) => {
            println!("\n[ERROR] GPU failed: {}", e);
            println!("Continuing with CPU benchmark only...\n");
            0.0
        }
    };

    // Test CPU
    let cpu_throughput = benchmark_cpu(num_keys);

    // Comparaison
    println!("\n{}", "=".repeat(60));
    println!("RESULTS COMPARISON");
    println!("{}", "=".repeat(60));
    println!("CPU (Rayon):    {:>12.2} keys/sec", cpu_throughput);

    if gpu_throughput > 0.0 {
        println!("GPU (CUDA):     {:>12.2} keys/sec", gpu_throughput);
        let speedup = gpu_throughput / cpu_throughput;
        println!("\nSpeedup:        {:>12.2}x", speedup);

        if speedup > 1.0 {
            println!("\n✓ GPU is {} times faster!", speedup);
        } else {
            println!("\n✗ CPU is faster (overhead from GPU transfers)");
        }
    }

    println!("{}\n", "=".repeat(60));

    Ok(())
}
