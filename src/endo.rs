// Clés dérivées d'une même clé privée k par les symétries de secp256k1.
//
// Le GPU calcule k*G = (x, y) une seule fois, puis en tire 6 clés publiques :
//   (beta^i * x, +-y)  <->  clés privées  +-lambda^i * k mod n
// Ce module reconstruit la clé privée correspondant à chaque indice (même ordre que
// dans kernels/secp256k1_kernel.cu) :
//   0: k   1: lambda*k   2: lambda^2*k   3: -k   4: -lambda*k   5: -lambda^2*k

use secp256k1::{Scalar, SecretKey};

pub const NUM_DERIVED: usize = 6;

/// lambda : racine cubique de l'unité mod n, associée à beta côté GPU
/// (lambda * (x, y) = (beta * x, y))
const LAMBDA: [u8; 32] = [
    0x53, 0x63, 0xad, 0x4c, 0xc0, 0x5c, 0x30, 0xe0, 0xa5, 0x26, 0x1c, 0x02, 0x88, 0x12, 0x64, 0x5a,
    0x12, 0x2e, 0x22, 0xea, 0x20, 0x81, 0x66, 0x78, 0xdf, 0x02, 0x96, 0x7c, 0x1b, 0x23, 0xbd, 0x72,
];

/// Clé privée dérivée numéro `index` (0..6) de la clé `private_key`
pub fn derived_private_key(private_key: &SecretKey, index: usize) -> SecretKey {
    assert!(index < NUM_DERIVED);
    let lambda = Scalar::from_be_bytes(LAMBDA).expect("lambda < n");

    let mut key = *private_key;
    for _ in 0..index % 3 {
        key = key.mul_tweak(&lambda).expect("lambda * k != 0");
    }
    if index >= 3 {
        key = key.negate();
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use secp256k1::{PublicKey, Secp256k1};

    #[test]
    fn derived_keys_share_coordinates() {
        let secp = Secp256k1::new();
        let key = SecretKey::from_slice(&[0x42; 32]).unwrap();
        let base = PublicKey::from_secret_key(&secp, &key).serialize_uncompressed();

        for index in 0..NUM_DERIVED {
            let derived = derived_private_key(&key, index);
            let public = PublicKey::from_secret_key(&secp, &derived).serialize_uncompressed();

            // L'endomorphisme conserve y (au signe près) ; l'opposé conserve x
            let same_y = public[33..] == base[33..];
            assert_eq!(same_y, index < 3, "index {}", index);
            assert_eq!(public[1..33] == base[1..33], index % 3 == 0, "index {}", index);
        }

        // lambda^3 = 1
        let lambda = Scalar::from_be_bytes(LAMBDA).unwrap();
        let cubed = key.mul_tweak(&lambda).unwrap().mul_tweak(&lambda).unwrap().mul_tweak(&lambda).unwrap();
        assert_eq!(cubed, key);
    }
}
