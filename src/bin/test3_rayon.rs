use secp256k1::{Secp256k1, SecretKey, PublicKey};
use sha2::{Sha256, Digest};
use ripemd::Ripemd160;
use bs58;
use rand::rngs::OsRng;
use std::time::Instant;
use std::io::{self, Write};
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

fn sequential(iterations: usize) -> f64 {
    println!("=== SEQUENTIAL (1 thread) ===\n");

    let start = Instant::now();

    for i in 0..iterations {
        let _address = create_p2pkh_address();

        if (i + 1) % 100000 == 0 {
            print!("\rGenerated {} addresses...", i + 1);
            io::stdout().flush().unwrap();
        }
    }

    let duration = start.elapsed();
    let throughput = iterations as f64 / duration.as_secs_f64();

    println!("\rGenerated {} addresses in {:.2}s", iterations, duration.as_secs_f64());
    println!("THROUGHPUT: {:.2} wallets/sec\n", throughput);

    throughput
}

fn parallel_rayon(iterations: usize) -> f64 {
    println!("=== PARALLEL (Rayon) ===\n");

    let start = Instant::now();

    let num_threads = rayon::current_num_threads();
    let chunk_size = (iterations + num_threads - 1) / num_threads;

    let _addresses: Vec<String> = (0..iterations)
        .into_par_iter()
        .chunks(chunk_size)
        .flat_map(|chunk| {
            chunk.into_iter().map(|_i| {
                create_p2pkh_address()
            }).collect::<Vec<_>>()
        })
        .collect();

    let duration = start.elapsed();
    let throughput = iterations as f64 / duration.as_secs_f64();

    println!("Generated {} addresses in {:.2}s", iterations, duration.as_secs_f64());
    println!("Threads used: {}", num_threads);
    println!("THROUGHPUT: {:.2} wallets/sec\n", throughput);

    throughput
}

fn main() {
    let mut input = String::new();
    println!("Enter number of wallet generations to benchmark: ");
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut input).unwrap();
    let iterations: usize = input.trim().parse().expect("Please enter a valid number");

    println!("\n");

    let seq_throughput = sequential(iterations);
    let par_throughput = parallel_rayon(iterations);

    let speedup = par_throughput / seq_throughput;

    println!("{:-^60}", "");
    println!("RESULTS:");
    println!("Sequential:  {:.2} wallets/sec", seq_throughput);
    println!("Parallel:    {:.2} wallets/sec", par_throughput);
    println!("SPEEDUP:     {:.2}x faster!", speedup);
    println!("{:-^60}\n", "");
}
