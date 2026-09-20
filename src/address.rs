// Adresses Bitcoin P2PKH : Base58Check rapide pour un payload fixe de 25 octets

use sha2::{Digest, Sha256};

pub const BASE58_ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Adresse Base58 dans un tampon fixe, sans allocation (25 octets -> 35 caractères max)
pub struct Address {
    buf: [u8; 35],
    len: usize,
}

impl Address {
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.len]).expect("base58 is ASCII")
    }
}

/// Base58 spécialisé pour 25 octets : le nombre de 200 bits tient dans 7 limbs u32,
/// et chaque division par 58^5 sort 5 caractères d'un coup (7 divisions longues au
/// lieu de ~35 pour l'algorithme générique octet par octet).
pub fn base58_encode_25(data: &[u8; 25]) -> Address {
    const POW5: u64 = 58 * 58 * 58 * 58 * 58;

    // Big-endian : limbs[0] = octet de poids fort, puis 6 mots de 32 bits
    let mut limbs = [0u32; 7];
    limbs[0] = data[0] as u32;
    for i in 0..6 {
        limbs[i + 1] = u32::from_be_bytes(data[1 + 4 * i..5 + 4 * i].try_into().unwrap());
    }

    // Chiffres base58, poids faible en premier (58^35 > 2^200)
    let mut digits = [0u8; 35];
    for chunk in digits.chunks_mut(5) {
        let mut rem = 0u64;
        for limb in limbs.iter_mut() {
            let cur = (rem << 32) | *limb as u64;
            *limb = (cur / POW5) as u32;
            rem = cur % POW5;
        }
        for digit in chunk {
            *digit = (rem % 58) as u8;
            rem /= 58;
        }
    }

    let mut num_digits = 35;
    while num_digits > 0 && digits[num_digits - 1] == 0 {
        num_digits -= 1;
    }

    // Chaque octet nul en tête devient un '1'
    let zeros = data.iter().take_while(|&&b| b == 0).count();

    let mut out = Address { buf: [b'1'; 35], len: zeros + num_digits };
    for i in 0..num_digits {
        out.buf[zeros + i] = BASE58_ALPHABET[digits[num_digits - 1 - i] as usize];
    }
    out
}

/// Adresse P2PKH (version 0x00) à partir du HASH160 de la clé publique
pub fn p2pkh_address(pubkey_hash: &[u8; 20]) -> Address {
    let mut full = [0u8; 25];
    full[1..21].copy_from_slice(pubkey_hash);
    let checksum = Sha256::digest(Sha256::digest(&full[..21]));
    full[21..].copy_from_slice(&checksum[..4]);
    base58_encode_25(&full)
}

// ---- Préfixe d'adresse -> intervalles de HASH160 (pour filtrer sur le GPU) ----

/// Entier de 256 bits, limbs little-endian
type U256 = [u32; 8];

/// a = a * m + add ; retourne false en cas de dépassement
fn mul_add_small(a: &mut U256, m: u32, add: u32) -> bool {
    let mut carry = add as u64;
    for limb in a.iter_mut() {
        carry += *limb as u64 * m as u64;
        *limb = carry as u32;
        carry >>= 32;
    }
    carry == 0
}

fn sub_one(a: &mut U256) {
    for limb in a.iter_mut() {
        let (v, borrow) = limb.overflowing_sub(1);
        *limb = v;
        if !borrow {
            break;
        }
    }
}

fn cmp(a: &U256, b: &U256) -> std::cmp::Ordering {
    a.iter().rev().cmp(b.iter().rev())
}

/// 2^bit
fn pow2(bit: usize) -> U256 {
    let mut r = [0u32; 8];
    r[bit / 32] = 1 << (bit % 32);
    r
}

/// Octets 1..21 du payload de 25 octets (le HASH160), c'est-à-dire bits 32..192
fn payload_to_hash160(n: &U256) -> [u8; 20] {
    let mut out = [0u8; 20];
    for i in 0..5 {
        out[16 - 4 * i..20 - 4 * i].copy_from_slice(&n[1 + i].to_be_bytes());
    }
    out
}

/// Intervalles [lo, hi] (bornes incluses) de HASH160 dont l'adresse P2PKH commence par `prefix`.
///
/// Une adresse est l'écriture en base 58 du payload N = version | HASH160 | checksum, donc
/// « commence par P » <=> N est dans [P * 58^j, (P+1) * 58^j) pour une des longueurs possibles.
/// Le checksum (4 octets de poids faible) est inconnu : les intervalles sont exacts sauf
/// pour un HASH160 égal à une borne, où seule une partie des checksums convient. Le filtre
/// est donc légèrement large, jamais trop étroit : à confirmer avec l'adresse réelle.
pub fn prefix_hash160_ranges(prefix: &str) -> Result<Vec<([u8; 20], [u8; 20])>, String> {
    let ones = prefix.bytes().take_while(|&c| c == b'1').count();
    if ones == 0 {
        return Err("a P2PKH address always starts with '1'".to_string());
    }
    if ones > 20 {
        return Err("prefix has too many leading '1'".to_string());
    }
    let rest = &prefix.as_bytes()[ones..];

    // Valeur du reste du préfixe en base 58
    let mut value: U256 = [0; 8];
    for &c in rest {
        let digit = BASE58_ALPHABET
            .iter()
            .position(|&a| a == c)
            .ok_or_else(|| format!("'{}' is not a Base58 character (0, O, I and l are excluded)", c as char))?;
        if !mul_add_small(&mut value, 58, digit as u32) {
            return Err("prefix too long".to_string());
        }
    }

    // Chaque '1' en tête est un octet nul en tête du payload (le premier est l'octet de version)
    let upper = {
        let mut b = pow2(8 * (25 - ones));
        sub_one(&mut b);
        b
    };
    if rest.is_empty() {
        // Au moins `ones` octets nuls : tout le bas de l'espace
        return Ok(vec![([0u8; 20], payload_to_hash160(&upper))]);
    }
    // Exactement `ones` octets nuls (le caractère suivant n'est pas '1')
    let lower = pow2(8 * (24 - ones));

    let mut ranges = Vec::new();
    let mut lo = value;
    let mut hi = value;
    let mut hi_valid = mul_add_small(&mut hi, 1, 1); // hi = value + 1
    // A chaque tour, un chiffre de plus derrière le préfixe : [value * 58^j, (value+1) * 58^j - 1]
    for _ in rest.len()..=35 {
        if cmp(&lo, &upper).is_gt() {
            break;
        }
        let mut last = if hi_valid { hi } else { [u32::MAX; 8] };
        if hi_valid {
            sub_one(&mut last);
        }

        let range_lo = if cmp(&lo, &lower).is_lt() { lower } else { lo };
        let range_hi = if cmp(&last, &upper).is_gt() { upper } else { last };
        if cmp(&range_lo, &range_hi).is_le() {
            ranges.push((payload_to_hash160(&range_lo), payload_to_hash160(&range_hi)));
        }

        if !mul_add_small(&mut lo, 58, 0) {
            break;
        }
        hi_valid = hi_valid && mul_add_small(&mut hi, 58, 0);
    }

    if ranges.is_empty() {
        return Err(format!("no P2PKH address can start with \"{}\"", prefix));
    }
    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_ranges_match_real_addresses() {
        for prefix in ["1", "11", "1A", "12", "1z", "1Ab", "1Q2"] {
            let ranges = prefix_hash160_ranges(prefix).unwrap();
            let mut hits = 0;

            let mut hash = [0u8; 20];
            for round in 0..200_000u32 {
                let next = Sha256::digest([&hash[..], &round.to_le_bytes()].concat());
                hash.copy_from_slice(&next[..20]);
                if round % 7 == 0 {
                    hash[0] = 0; // adresses en "11..."
                }

                let starts = p2pkh_address(&hash).as_str().starts_with(prefix);
                let in_range = ranges.iter().any(|(lo, hi)| *lo <= hash && hash <= *hi);
                let on_bound = ranges.iter().any(|(lo, hi)| *lo == hash || *hi == hash);

                // Jamais trop étroit ; large seulement sur une borne
                assert!(!starts || in_range, "{}: {} missed", prefix, hex::encode(hash));
                assert!(starts || !in_range || on_bound, "{}: {} wrongly kept", prefix, hex::encode(hash));
                hits += starts as u32;
            }
            assert!(hits > 0, "{}: test never exercised a match", prefix);
        }

        assert!(prefix_hash160_ranges("1O").is_err());
        assert!(prefix_hash160_ranges("3A").is_err());
    }

    #[test]
    fn matches_bs58_crate() {
        // Cas limites (zéros en tête) + valeurs pseudo-aléatoires
        let mut hash = [0u8; 20];
        for round in 0..1000u32 {
            let mut full = vec![0u8];
            full.extend(hash);
            let checksum = Sha256::digest(Sha256::digest(&full));
            full.extend(&checksum[..4]);
            assert_eq!(p2pkh_address(&hash).as_str(), bs58::encode(&full).into_string());

            let next = Sha256::digest([&hash[..], &round.to_le_bytes()].concat());
            hash.copy_from_slice(&next[..20]);
            if round % 3 == 0 {
                hash[..(round as usize / 3) % 6].fill(0);
            }
        }
    }
}
