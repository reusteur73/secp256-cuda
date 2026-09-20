// Test pour valider les opérations arithmétiques modulo p

fn main() {
    // Test multiplication: 2 * 3 mod p = 6
    // Test addition: 1 + 2 mod p = 3
    // Test avec les vraies valeurs de secp256k1

    // p = 2^256 - 2^32 - 977 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

    let p: u128 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFD - 0xFFFFFFFF; // simplifié pour test

    // Test simple: 2 * 3 = 6
    let a: u128 = 2;
    let b: u128 = 3;
    let expected: u128 = 6;
    let result = (a * b) % p;

    println!("2 * 3 mod p = {} (expected {})", result, expected);

    // Pour secp256k1, test avec vraie multiplication des points
    // G = (Gx, Gy) où:
    // Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BCCDC1873349838D7DF14A7D
    // Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10

    use secp256k1::{Secp256k1, SecretKey};

    let secp = Secp256k1::new();

    // Test k=1 (should give G)
    let secret1 = SecretKey::from_slice(&[0u8; 32]).unwrap_or_else(|_| {
        let mut bytes = [0u8; 32];
        bytes[31] = 1;
        SecretKey::from_slice(&bytes).unwrap()
    });

    let pubkey1 = secp256k1::PublicKey::from_secret_key(&secp, &secret1);
    let ser1 = pubkey1.serialize_uncompressed();

    println!("\nk=1:");
    println!("  X: {}", hex::encode(&ser1[1..33]));
    println!("  Y: {}", hex::encode(&ser1[33..65]));

    // Expected G values
    println!("\nExpected G:");
    println!("  Gx: 79be667ef9dcbbac55a06295ce870b07029bccdc1873349838d7df14a7d");
    println!("  Gy: 483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10");
}
