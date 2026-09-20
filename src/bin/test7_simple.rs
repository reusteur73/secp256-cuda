// Test ultra-simple: k=1 devrait retourner G

mod gpu {
    pub use utxo_loader::gpu::*;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ultra-Simple Test: k=1 should return G ===\n");

    // k = 1 (all zeros except last byte = 1)
    let mut k = [0u8; 32];
    k[31] = 1;

    // Expected G coordinates from secp256k1
    let gx_hex = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    let gy_hex = "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";

    println!("Expected G:");
    println!("  Gx: {}", gx_hex);
    println!("  Gy: {}", gy_hex);

    println!("\nGPU result:");
    let mut gpu_ctx = gpu::GpuContext::new()?;
    let (pub_x, pub_y) = gpu_ctx.derive_public_keys_batch(&k, 1)?;

    let gpu_x_hex = hex::encode(&pub_x);
    let gpu_y_hex = hex::encode(&pub_y);

    println!("  Gx: {}", gpu_x_hex);
    println!("  Gy: {}", gpu_y_hex);

    if gpu_x_hex == gx_hex && gpu_y_hex == gy_hex {
        println!("\n✓ SUCCESS! GPU produced correct G!");
    } else {
        println!("\n✗ FAIL - Results don't match");

        // Try byte reversal
        let mut gx_rev = pub_x.clone();
        gx_rev.reverse();
        let gx_rev_hex = hex::encode(&gx_rev);

        if gx_rev_hex == gx_hex {
            println!("\n  But reversed bytes match! Endianness issue.");
        }
    }

    Ok(())
}
