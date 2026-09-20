use secp256k1::{Secp256k1, SecretKey, PublicKey};
use sha2::{Sha256, Digest};
use ripemd::Ripemd160;
use bs58;
use rand::rngs::OsRng;
use std::time::Instant;
use std::io::{self, Write};

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

fn main() {
    let mut input = String::new();
    println!("Enter number of wallet generations to benchmark: ");
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut input).unwrap();
    let iterations: usize = input.trim().parse().expect("Please enter a valid number");

    println!("\n=== ORIGINAL (recrée Secp256k1 à chaque fois) ===\n");
    benchmark_original(iterations);

    println!("\n=== OPTIMIZED (Secp256k1 réutilisé) ===\n");
    benchmark_optimized(iterations);
}

fn benchmark_original(iterations: usize) {
    let mut times = [0.0; 5];
    let mut total_time = 0.0;
    let mut rng = OsRng;

    let benchmark_start = Instant::now();

    for i in 0..iterations {
        let total_start = Instant::now();

        // Step 1: Create SecretKey
        let step_start = Instant::now();
        let secp = Secp256k1::new();
        let secret_key = SecretKey::new(&mut rng);
        times[0] += step_start.elapsed().as_secs_f64() * 1000.0;

        // Step 2: Derive PublicKey
        let step_start = Instant::now();
        let public_key = PublicKey::from_secret_key(&secp, &secret_key);
        let pubkey_bytes = public_key.serialize();
        times[1] += step_start.elapsed().as_secs_f64() * 1000.0;

        // Step 3: Hash public key (SHA256 + RIPEMD160)
        let step_start = Instant::now();
        let pubkey_hash = ripemd160(&sha256(&pubkey_bytes));
        times[2] += step_start.elapsed().as_secs_f64() * 1000.0;

        // Step 4: Add version byte and prepare payload
        let step_start = Instant::now();
        let mut payload = vec![0x00];
        payload.extend(pubkey_hash);
        times[3] += step_start.elapsed().as_secs_f64() * 1000.0;

        // Step 5: Base58Check encode
        let step_start = Instant::now();
        let _address = base58check_encode(&payload);
        times[4] += step_start.elapsed().as_secs_f64() * 1000.0;

        total_time += total_start.elapsed().as_secs_f64() * 1000.0;

        if (i + 1) % 100 == 0 {
            print!("\rGenerated {} wallets...", i + 1);
            io::stdout().flush().unwrap();
        }
    }

    let benchmark_duration = benchmark_start.elapsed();
    println!("\rGenerated {} wallets in {:.2}s\n", iterations, benchmark_duration.as_secs_f64());

    println!("{:-^60}", " AVERAGE TIMES PER OPERATION ");
    println!("[1] Create SecretKey + Secp256k1: {:.4} ms ({:.1}%)", times[0] / iterations as f64, (times[0] / total_time) * 100.0);
    println!("[2] Derive PublicKey:             {:.4} ms ({:.1}%)", times[1] / iterations as f64, (times[1] / total_time) * 100.0);
    println!("[3] SHA256 + RIPEMD160:           {:.4} ms ({:.1}%)", times[2] / iterations as f64, (times[2] / total_time) * 100.0);
    println!("[4] Prepare payload:              {:.4} ms ({:.1}%)", times[3] / iterations as f64, (times[3] / total_time) * 100.0);
    println!("[5] Base58Check encode:           {:.4} ms ({:.1}%)", times[4] / iterations as f64, (times[4] / total_time) * 100.0);

    println!("\n{:-^60}", "");
    println!("AVERAGE TOTAL TIME: {:.4} ms", total_time / iterations as f64);
    println!("TOTAL THROUGHPUT:   {:.2} wallets/sec", iterations as f64 / benchmark_duration.as_secs_f64());
    println!("{:-^60}", "");
}

fn benchmark_optimized(iterations: usize) {
    let mut times = [0.0; 5];
    let mut total_time = 0.0;

    // ✅ Créer Secp256k1 UNE SEULE FOIS
    let secp = Secp256k1::new();
    let mut rng = OsRng;

    println!("Running {} wallet generations (Secp256k1 réutilisé)...\n", iterations);

    let benchmark_start = Instant::now();

    for i in 0..iterations {
        let total_start = Instant::now();

        // Step 1: Create SecretKey only
        let step_start = Instant::now();
        let secret_key = SecretKey::new(&mut rng);
        times[0] += step_start.elapsed().as_secs_f64() * 1000.0;

        // Step 2: Derive PublicKey (réutilise secp)
        let step_start = Instant::now();
        let public_key = PublicKey::from_secret_key(&secp, &secret_key);
        let pubkey_bytes = public_key.serialize();
        times[1] += step_start.elapsed().as_secs_f64() * 1000.0;

        // Step 3: Hash public key (SHA256 + RIPEMD160)
        let step_start = Instant::now();
        let pubkey_hash = ripemd160(&sha256(&pubkey_bytes));
        times[2] += step_start.elapsed().as_secs_f64() * 1000.0;

        // Step 4: Add version byte and prepare payload
        let step_start = Instant::now();
        let mut payload = vec![0x00];
        payload.extend(pubkey_hash);
        times[3] += step_start.elapsed().as_secs_f64() * 1000.0;

        // Step 5: Base58Check encode
        let step_start = Instant::now();
        let _address = base58check_encode(&payload);
        times[4] += step_start.elapsed().as_secs_f64() * 1000.0;

        total_time += total_start.elapsed().as_secs_f64() * 1000.0;

        if (i + 1) % 100 == 0 {
            print!("\rGenerated {} wallets...", i + 1);
            io::stdout().flush().unwrap();
        }
    }

    let benchmark_duration = benchmark_start.elapsed();
    println!("\rGenerated {} wallets in {:.2}s\n", iterations, benchmark_duration.as_secs_f64());

    println!("{:-^60}", " AVERAGE TIMES PER OPERATION ");
    println!("[1] Create SecretKey:             {:.4} ms ({:.1}%)", times[0] / iterations as f64, (times[0] / total_time) * 100.0);
    println!("[2] Derive PublicKey:             {:.4} ms ({:.1}%)", times[1] / iterations as f64, (times[1] / total_time) * 100.0);
    println!("[3] SHA256 + RIPEMD160:           {:.4} ms ({:.1}%)", times[2] / iterations as f64, (times[2] / total_time) * 100.0);
    println!("[4] Prepare payload:              {:.4} ms ({:.1}%)", times[3] / iterations as f64, (times[3] / total_time) * 100.0);
    println!("[5] Base58Check encode:           {:.4} ms ({:.1}%)", times[4] / iterations as f64, (times[4] / total_time) * 100.0);

    println!("\n{:-^60}", "");
    println!("AVERAGE TOTAL TIME: {:.4} ms", total_time / iterations as f64);
    println!("TOTAL THROUGHPUT:   {:.2} wallets/sec", iterations as f64 / benchmark_duration.as_secs_f64());
    println!("{:-^60}", "");
}
