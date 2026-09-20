// CUDA Kernel pour secp256k1 : dérivation de clés publiques (k*G), HASH160 et filtre
//
// Méthode : fenêtres à base fixe, digits signés. La table (construite côté CPU, voir
// src/gpu.rs) contient d * 2^(W*w) * G en affine pour chaque fenêtre w et d dans 1..=2^(W-1).
// k*G = somme d'une entrée par fenêtre -> aucune duplication de point, seulement
// des additions mixtes Jacobien+affine, puis UNE inversion par thread (astuce de
// Montgomery sur KEYS_PER_THREAD clés).
//
// Représentation : scalaires, table et sorties en 8 limbs de 32 bits little-endian (BN256) ;
// calculs dans le corps en 10 limbs de 26 bits (FE), voir plus bas.

#include <stdint.h>

// Nombre de clés par thread GPU (une inversion modulaire par thread).
// Défini par build.rs (-DKEYS_PER_THREAD=N), qui transmet la même valeur au code Rust.
#ifndef KEYS_PER_THREAD
#define KEYS_PER_THREAD 32
#endif

// ============ Entiers 256 bits canoniques (mod p = 2^256 - 2^32 - 977) ============

typedef struct {
    uint32_t v[8];
} BN256;

// Si carry ou x >= p : x = x - p, c'est-à-dire x + (2^32 + 977) mod 2^256
__device__ void bn_normalize(BN256* x, uint32_t carry) {
    BN256 t;
    uint64_t acc = (uint64_t)x->v[0] + 977;
    t.v[0] = (uint32_t)acc; acc >>= 32;
    acc += (uint64_t)x->v[1] + 1;
    t.v[1] = (uint32_t)acc; acc >>= 32;
    for (int i = 2; i < 8; i++) {
        acc += x->v[i];
        t.v[i] = (uint32_t)acc; acc >>= 32;
    }
    if (carry | (uint32_t)acc) *x = t;
}

// Réduction d'un produit 512 bits : 2^256 = 2^32 + 977 (mod p)  =>  r = lo + hi * (2^32 + 977)
__device__ void bn_reduce(BN256* dst, const uint32_t* t) {
    BN256 r;
    uint64_t acc = 0;
    uint32_t prev = 0;
    for (int i = 0; i < 8; i++) {
        acc += (uint64_t)t[i] + (uint64_t)t[i + 8] * 977 + prev;
        r.v[i] = (uint32_t)acc;
        acc >>= 32;
        prev = t[i + 8];
    }
    uint64_t o = acc + prev;  // dépassement (limb 8), < 2^34

    // Second repli : r += o * (2^32 + 977)
    uint64_t m = o * 977;
    acc = (uint64_t)r.v[0] + (uint32_t)m;
    r.v[0] = (uint32_t)acc; acc >>= 32;
    acc += (uint64_t)r.v[1] + (m >> 32) + (uint32_t)o;
    r.v[1] = (uint32_t)acc; acc >>= 32;
    acc += (uint64_t)r.v[2] + (o >> 32);
    r.v[2] = (uint32_t)acc; acc >>= 32;
    for (int i = 3; i < 8; i++) {
        acc += r.v[i];
        r.v[i] = (uint32_t)acc; acc >>= 32;
    }

    bn_normalize(&r, (uint32_t)acc);
    *dst = r;
}

// Multiplication 8x32 : n'est plus utilisée par les kernels, gardée comme référence
// pour kernels/bench_mul.cu
__device__ void bn_mul(BN256* dst, const BN256* a, const BN256* b) {
    uint32_t t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;

    // Produit 512 bits
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            carry += (uint64_t)a->v[i] * b->v[j] + t[i + j];
            t[i + j] = (uint32_t)carry;
            carry >>= 32;
        }
        t[i + 8] = (uint32_t)carry;
    }

    bn_reduce(dst, t);
}

// ============ Corps fini mod p : 10 limbs de 26 bits ============
//
// Même représentation que libsecp256k1 en 32 bits. Avantages sur GPU (pas de drapeau de
// retenue) : les colonnes d'un produit s'accumulent dans 64 bits SANS propagation de retenue,
// et additions / soustractions se font limb par limb, sans retenue non plus.
// Mesuré avec kernels/bench_mul.cu : +22 % sur la multiplication par rapport à 8x32.
//
// Contrepartie : les limbs ne sont pas toujours normalisés. La « magnitude » m d'une valeur
// borne ses limbs : n[i] <= 2*m*(2^26-1). Une multiplication accepte des magnitudes dont le
// produit est <= 64 (colonnes < 2^64) et rend une magnitude 1. Les magnitudes sont suivies
// à la main dans les commentaires « mag ».

typedef struct {
    uint32_t n[10];
} FE;

#define FE_M26 0x3FFFFFFu

__device__ void fe_set_one(FE* r) {
    r->n[0] = 1;
    for (int i = 1; i < 10; i++) r->n[i] = 0;
}

// Depuis la forme canonique 8x32 (mag 1)
__device__ void fe_from_limbs(FE* r, const uint32_t* v) {
    for (int i = 0; i < 10; i++) {
        int pos = 26 * i;
        int limb = pos >> 5, sh = pos & 31;
        uint64_t x = v[limb] >> sh;
        if (limb < 7) x |= (uint64_t)v[limb + 1] << (32 - sh);
        r->n[i] = (uint32_t)x & FE_M26;
    }
}

// Vers la forme canonique 8x32 (réduction complète). Entrée : mag 1.
__device__ void fe_to_bn(BN256* r, const FE* a) {
    uint32_t t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;
    uint64_t acc = 0;
    int bits = 0, out = 0;
    for (int i = 0; i < 10; i++) {
        acc += (uint64_t)a->n[i] << bits;
        bits += (i < 9) ? 26 : 32;
        while (bits >= 32) {
            t[out++] = (uint32_t)acc;
            acc >>= 32;
            bits -= 32;
        }
    }
    t[out] = (uint32_t)acc;
    bn_reduce(r, t);
}

// r = -a pour a de magnitude <= m ; résultat de magnitude m + 1
__device__ void fe_negate(FE* r, const FE* a, uint32_t m) {
    const uint32_t k = 2 * (m + 1);
    r->n[0] = 0x3FFFC2Fu * k - a->n[0];
    r->n[1] = 0x3FFFFBFu * k - a->n[1];
    for (int i = 2; i < 9; i++) r->n[i] = 0x3FFFFFFu * k - a->n[i];
    r->n[9] = 0x03FFFFFu * k - a->n[9];
}

// r += a ; les magnitudes s'additionnent
__device__ void fe_add(FE* r, const FE* a) {
    for (int i = 0; i < 10; i++) r->n[i] += a->n[i];
}

// Réduit les 19 colonnes d'un produit vers 10 limbs (mag 1)
__device__ void fe_reduce_columns(FE* r, const uint64_t* c) {
    // Partie haute (colonnes 10..18) en limbs de 26 bits ; h[9] garde le reste
    uint64_t h[10];
    uint64_t carry = 0;
    for (int k = 0; k < 9; k++) {
        carry += c[10 + k];
        h[k] = carry & FE_M26;
        carry >>= 26;
    }
    h[9] = carry;

    // 2^260 = 2^36 + 0x3D10 (mod p)  =>  colonne k+10 -> 0x3D10 en k, 0x400 en k+1
    uint64_t d[11];
    d[0] = c[0] + h[0] * 0x3D10u;
    for (int k = 1; k < 10; k++) d[k] = c[k] + h[k] * 0x3D10u + h[k - 1] * 0x400u;
    d[10] = h[9] * 0x400u;

    carry = 0;
    for (int k = 0; k < 9; k++) {
        carry += d[k];
        r->n[k] = (uint32_t)carry & FE_M26;
        carry >>= 26;
    }
    carry += d[9];
    r->n[9] = (uint32_t)carry & 0x3FFFFFu;         // 22 bits : 9*26 + 22 = 256
    uint64_t top = (carry >> 22) + (d[10] << 4);   // dépassement, en unités de 2^256

    // 2^256 = 2^32 + 0x3D1 (mod p)
    carry = (uint64_t)r->n[0] + top * 0x3D1u;
    r->n[0] = (uint32_t)carry & FE_M26;
    carry >>= 26;
    carry += (uint64_t)r->n[1] + (top << 6);
    r->n[1] = (uint32_t)carry & FE_M26;
    carry >>= 26;
    for (int k = 2; k < 9; k++) {
        carry += r->n[k];
        r->n[k] = (uint32_t)carry & FE_M26;
        carry >>= 26;
    }
    r->n[9] += (uint32_t)carry;
}

// r = a * b ; mag(a) * mag(b) <= 64 ; résultat mag 1
__device__ void fe_mul(FE* r, const FE* a, const FE* b) {
    uint64_t c[19];
    for (int k = 0; k < 19; k++) c[k] = 0;
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            c[i + j] += (uint64_t)a->n[i] * b->n[j];
    fe_reduce_columns(r, c);
}

// r = a^2 ; mag(a) <= 8 ; résultat mag 1. 55 produits au lieu de 100.
__device__ void fe_sqr(FE* r, const FE* a) {
    uint64_t c[19];
    for (int k = 0; k < 19; k++) c[k] = 0;
    for (int i = 0; i < 10; i++) {
        c[2 * i] += (uint64_t)a->n[i] * a->n[i];
        uint32_t twice = a->n[i] * 2;   // <= 2^31 pour mag <= 8
        for (int j = i + 1; j < 10; j++)
            c[i + j] += (uint64_t)twice * a->n[j];
    }
    fe_reduce_columns(r, c);
}

__device__ void fe_sqr_n(FE* x, int n) {
    for (int i = 0; i < n; i++) fe_sqr(x, x);
}

// (noinline : sinon chaque site d'appel duplique tout le code, le PTX explose et le
// driver met des minutes à charger le module)
// Fermat: a^-1 = a^(p-2) mod p, chaîne d'additions de libsecp256k1 (255 S + 15 M). mag 1 -> mag 1
__device__ __noinline__ void fe_inv(FE* dst, const FE* a) {
    FE x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;

    x2 = *a;    fe_sqr_n(&x2, 1);     fe_mul(&x2, &x2, a);
    x3 = x2;    fe_sqr_n(&x3, 1);     fe_mul(&x3, &x3, a);
    x6 = x3;    fe_sqr_n(&x6, 3);     fe_mul(&x6, &x6, &x3);
    x9 = x6;    fe_sqr_n(&x9, 3);     fe_mul(&x9, &x9, &x3);
    x11 = x9;   fe_sqr_n(&x11, 2);    fe_mul(&x11, &x11, &x2);
    x22 = x11;  fe_sqr_n(&x22, 11);   fe_mul(&x22, &x22, &x11);
    x44 = x22;  fe_sqr_n(&x44, 22);   fe_mul(&x44, &x44, &x22);
    x88 = x44;  fe_sqr_n(&x88, 44);   fe_mul(&x88, &x88, &x44);
    x176 = x88; fe_sqr_n(&x176, 88);  fe_mul(&x176, &x176, &x88);
    x220 = x176; fe_sqr_n(&x220, 44); fe_mul(&x220, &x220, &x44);
    x223 = x220; fe_sqr_n(&x223, 3);  fe_mul(&x223, &x223, &x3);

    t = x223;
    fe_sqr_n(&t, 23); fe_mul(&t, &t, &x22);
    fe_sqr_n(&t, 5);  fe_mul(&t, &t, a);
    fe_sqr_n(&t, 3);  fe_mul(&t, &t, &x2);
    fe_sqr_n(&t, 2);  fe_mul(dst, &t, a);
}

// ============ Elliptic Curve Points ============

// Point en coordonnées jacobiennes (x = X/Z^2, y = Y/Z^3).
// Magnitudes maximales entre deux additions : x mag 6, y mag 3, z mag 1.
typedef struct {
    FE x, y, z;
    bool is_inf;
} ECPointJ;

// R = R + Q (ou R - Q si negate), Q affine lu depuis la table (16 limbs 8x32 : x puis y).
// Pas de cas exceptionnel (R == +-Q) en pratique : la somme signée des fenêtres basses
// est toujours plus petite en valeur absolue que la contribution de la fenêtre w ; seule
// une poignée de clés collées à n (probabilité ~2^-256) pourrait y échapper modulo n.
__device__ __noinline__ void point_add_mixed(ECPointJ* R, const uint32_t* q, bool negate) {
    FE qx, qy;
    fe_from_limbs(&qx, q);            // mag 1
    fe_from_limbs(&qy, q + 8);        // mag 1
    if (negate) fe_negate(&qy, &qy, 1);   // -Q = (x, -y), mag 2

    if (R->is_inf) {
        R->x = qx;
        R->y = qy;
        fe_set_one(&R->z);
        R->is_inf = false;
        return;
    }

    FE z2, u2, s2, h, r, h2, h3, v, x3, t;

    fe_sqr(&z2, &R->z);
    fe_mul(&u2, &qx, &z2);            // U2 = Qx*Z^2            mag 1
    fe_mul(&s2, &qy, &z2);
    fe_mul(&s2, &s2, &R->z);          // S2 = Qy*Z^3            mag 1
    fe_negate(&h, &R->x, 6);
    fe_add(&h, &u2);                  // H = U2 - X             mag 8
    fe_negate(&r, &R->y, 3);
    fe_add(&r, &s2);                  // r = S2 - Y             mag 5
    fe_sqr(&h2, &h);                  //                        mag 1
    fe_mul(&h3, &h2, &h);             // H^3                    mag 1
    fe_mul(&v, &R->x, &h2);           // V = X*H^2              mag 1

    // X3 = r^2 - H^3 - 2V
    fe_sqr(&x3, &r);                  //                        mag 1
    fe_negate(&t, &h3, 1);
    fe_add(&x3, &t);                  //                        mag 3
    t = v;
    fe_add(&t, &v);                   // 2V                     mag 2
    fe_negate(&t, &t, 2);
    fe_add(&x3, &t);                  //                        mag 6

    // Y3 = r*(V - X3) - Y*H^3
    fe_negate(&t, &x3, 6);
    fe_add(&t, &v);                   // V - X3                 mag 8
    fe_mul(&t, &t, &r);               // 8 * 5 <= 64            mag 1
    fe_mul(&h3, &h3, &R->y);          // Y*H^3                  mag 1
    fe_negate(&h3, &h3, 1);
    fe_add(&t, &h3);                  //                        mag 3

    R->x = x3;
    R->y = t;
    fe_mul(&R->z, &R->z, &h);         // Z3 = Z*H               mag 1
}

// k*G via la table à base fixe, avec des digits SIGNÉS : chaque fenêtre vaut d dans
// [-2^(W-1), 2^(W-1)] (une retenue passe à la fenêtre suivante quand le digit brut dépasse
// 2^(W-1)). La table ne stocke que |d| * 2^(W*w) * G, soit 2^(W-1) entrées par fenêtre :
// deux fois moins de mémoire, donc des fenêtres plus larges et moins d'additions par clé.
// La dernière fenêtre (256/W + 1 au total) absorbe la retenue finale.
__device__ __noinline__ void scalar_mult_base(ECPointJ* out, const BN256* k,
                               const uint32_t* table, uint32_t window_bits) {
    const uint32_t num_windows = 256 / window_bits + 1;
    const uint32_t half = 1u << (window_bits - 1);  // entrées par fenêtre
    const uint32_t mask = (half << 1) - 1;

    out->is_inf = true;
    uint32_t carry = 0;
    for (uint32_t w = 0; w < num_windows; w++) {
        uint32_t pos = w * window_bits;
        uint32_t limb = pos >> 5;
        uint32_t sh = pos & 31;
        uint32_t raw = 0;
        if (limb < 8) {
            raw = k->v[limb] >> sh;
            if (sh + window_bits > 32 && limb < 7) raw |= k->v[limb + 1] << (32 - sh);
            raw &= mask;
        }

        uint32_t d = raw + carry;
        bool negate = d > half;
        if (negate) d = (mask + 1) - d;           // digit = d - 2^W < 0, retenue de 1
        carry = negate ? 1 : 0;
        if (d == 0) continue;

        size_t index = (size_t)w * half + (d - 1);
        point_add_mixed(out, table + index * 16, negate);
    }
}

__device__ void bn_load_be(BN256* x, const unsigned char* bytes) {
    for (int i = 0; i < 8; i++) {
        const unsigned char* b = bytes + 28 - 4 * i;
        x->v[i] = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
                  ((uint32_t)b[2] << 8) | (uint32_t)b[3];
    }
}

__device__ void bn_store_be(unsigned char* bytes, const BN256* x) {
    for (int i = 0; i < 8; i++) {
        unsigned char* b = bytes + 28 - 4 * i;
        uint32_t v = x->v[i];
        b[0] = (unsigned char)(v >> 24);
        b[1] = (unsigned char)(v >> 16);
        b[2] = (unsigned char)(v >> 8);
        b[3] = (unsigned char)v;
    }
}


// ============ HASH160 = RIPEMD160(SHA256(clé publique compressée)) ============

__constant__ uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// RIPEMD160 : ordre des mots (R) et rotations (S), ligne gauche (1) et droite (2)
__constant__ uint8_t RMD_R1[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};
__constant__ uint8_t RMD_R2[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};
__constant__ uint8_t RMD_S1[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};
__constant__ uint8_t RMD_S2[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};
__constant__ uint32_t RMD_K1[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
__constant__ uint32_t RMD_K2[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};

__device__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

// SHA256 d'un message d'un seul bloc déjà paddé (16 mots big-endian)
__device__ void sha256_single_block(uint32_t* digest, const uint32_t* block) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = block[i];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    const uint32_t h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    const uint32_t h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
    uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;

    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    digest[0] = h0 + a; digest[1] = h1 + b; digest[2] = h2 + c; digest[3] = h3 + d;
    digest[4] = h4 + e; digest[5] = h5 + f; digest[6] = h6 + g; digest[7] = h7 + h;
}

__device__ uint32_t rmd_f(int round, uint32_t x, uint32_t y, uint32_t z) {
    switch (round) {
        case 0: return x ^ y ^ z;
        case 1: return (x & y) | (~x & z);
        case 2: return (x | ~y) ^ z;
        case 3: return (x & z) | (y & ~z);
        default: return x ^ (y | ~z);
    }
}

// RIPEMD160 d'un message d'un seul bloc déjà paddé (16 mots little-endian)
__device__ void ripemd160_single_block(uint32_t* digest, const uint32_t* x) {
    const uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
    uint32_t a1 = h0, b1 = h1, c1 = h2, d1 = h3, e1 = h4;
    uint32_t a2 = h0, b2 = h1, c2 = h2, d2 = h3, e2 = h4;

    for (int j = 0; j < 80; j++) {
        int round = j / 16;
        uint32_t t = rotl32(a1 + rmd_f(round, b1, c1, d1) + x[RMD_R1[j]] + RMD_K1[round], RMD_S1[j]) + e1;
        a1 = e1; e1 = d1; d1 = rotl32(c1, 10); c1 = b1; b1 = t;

        // La ligne droite utilise les fonctions dans l'ordre inverse
        t = rotl32(a2 + rmd_f(4 - round, b2, c2, d2) + x[RMD_R2[j]] + RMD_K2[round], RMD_S2[j]) + e2;
        a2 = e2; e2 = d2; d2 = rotl32(c2, 10); c2 = b2; b2 = t;
    }

    digest[0] = h1 + c1 + d2;
    digest[1] = h2 + d1 + e2;
    digest[2] = h3 + e1 + a2;
    digest[3] = h4 + a1 + b2;
    digest[4] = h0 + b1 + c2;
}

__device__ uint32_t bswap32(uint32_t v) {
    return (v >> 24) | ((v >> 8) & 0xFF00) | ((v << 8) & 0xFF0000) | (v << 24);
}

// HASH160 de la clé publique compressée (02/03 selon la parité de Y, puis X big-endian).
// Résultat en 5 mots BIG-endian : mêmes octets que le hash, et comparables comme un entier.
__device__ __noinline__ void hash160_words(uint32_t* out_be, const BN256* x, uint32_t y_parity) {
    // Message de 33 octets : prefix | X, avec le padding SHA256 (un seul bloc)
    uint32_t prefix = 0x02 | (y_parity & 1);
    uint32_t block[16];
    block[0] = (prefix << 24) | (x->v[7] >> 8);
    for (int i = 1; i < 8; i++) block[i] = (x->v[8 - i] << 24) | (x->v[7 - i] >> 8);
    block[8] = (x->v[0] << 24) | 0x00800000;
    for (int i = 9; i < 15; i++) block[i] = 0;
    block[15] = 33 * 8;

    uint32_t sha[8];
    sha256_single_block(sha, block);

    // Message de 32 octets pour RIPEMD160 (mots little-endian), avec son padding
    uint32_t rblock[16];
    for (int i = 0; i < 8; i++) rblock[i] = bswap32(sha[i]);
    rblock[8] = 0x00000080;
    for (int i = 9; i < 14; i++) rblock[i] = 0;
    rblock[14] = 32 * 8;
    rblock[15] = 0;

    uint32_t rmd[5];
    ripemd160_single_block(rmd, rblock);

    for (int i = 0; i < 5; i++) out_be[i] = bswap32(rmd[i]);
}

__device__ void hash160_store(unsigned char* out, const uint32_t* words_be) {
    for (int i = 0; i < 5; i++) {
        out[4 * i] = (unsigned char)(words_be[i] >> 24);
        out[4 * i + 1] = (unsigned char)(words_be[i] >> 16);
        out[4 * i + 2] = (unsigned char)(words_be[i] >> 8);
        out[4 * i + 3] = (unsigned char)words_be[i];
    }
}

// ============ Clés dérivées (symétries de secp256k1) ============
//
// A partir d'un seul k*G = (x, y), cinq autres clés publiques sont presque gratuites :
//   endomorphisme : (lambda^i * k) * G = (beta^i * x, y)     -> 1 multiplication par point
//   opposé        : (n - k) * G        = (x, p - y)          -> la parité de y s'inverse
// Ordre des 6 clés dérivées (doit correspondre à src/endo.rs) :
//   0: k   1: lambda*k   2: lambda^2*k   3: -k   4: -lambda*k   5: -lambda^2*k
// Chaque clé dérivée garde son propre HASH160 : c'est ce coût qui borne le gain.

#define MAX_DERIVED 6

// beta : racine cubique de l'unité mod p (8 limbs de 32 bits little-endian)
__constant__ uint32_t SECP256K1_BETA[8] = {
    0x719501ee, 0xc1396c28, 0x12f58995, 0x9cf04975,
    0xac3434e9, 0x6e64479e, 0x657c0710, 0x7ae96a2b
};

// HASH160 (mots big-endian) des clés dérivées d'un point affine. xs = {x, beta*x, beta^2*x}
// (seul xs[0] est lu sans endomorphisme). Retourne le nombre de clés dérivées : 1 ou 6.
__device__ unsigned int derived_hashes(uint32_t words[MAX_DERIVED][5], const BN256* xs,
                                     const BN256* y, bool endo) {
    uint32_t parity = y->v[0] & 1;
    if (!endo) {
        hash160_words(words[0], &xs[0], parity);
        return 1;
    }
    for (int i = 0; i < 3; i++) {
        hash160_words(words[i], &xs[i], parity);
        hash160_words(words[3 + i], &xs[i], parity ^ 1);  // p - y : parité opposée (p impair, y != 0)
    }
    return MAX_DERIVED;
}

// lo <= w <= hi, comparaison d'entiers de 160 bits (5 mots big-endian)
__device__ bool hash160_in_range(const uint32_t* w, const uint32_t* lo, const uint32_t* hi) {
    bool ge = true, le = true;
    for (int i = 0; i < 5; i++) {
        if (w[i] != lo[i]) { ge = w[i] > lo[i]; break; }
    }
    for (int i = 0; i < 5; i++) {
        if (w[i] != hi[i]) { le = w[i] < hi[i]; break; }
    }
    return ge && le;
}

// ============ Main Kernels ============

// Versions non inlinees pour les boucles par cle ci-dessous : ces boucles sont deroulees
// KEYS_PER_THREAD fois, et avec fe_mul inline la taille du PTX (donc le temps de chargement
// du module par le driver) grandit avec KEYS_PER_THREAD.
__device__ __noinline__ void fe_mul_call(FE* dst, const FE* a, const FE* b) { fe_mul(dst, a, b); }
__device__ __noinline__ void fe_sqr_call(FE* dst, const FE* a) { fe_sqr(dst, a); }
__device__ __noinline__ void fe_to_bn_call(BN256* dst, const FE* a) { fe_to_bn(dst, a); }

// Chaque thread traite KEYS_PER_THREAD clés consécutives à partir de `first`.
// En sortie, pour la clé j : out_x[3*j] = x, et si endo out_x[3*j+1] = beta*x,
// out_x[3*j+2] = beta^2*x ; out_y[j] = y. Coordonnées AFFINES canoniques (une seule
// inversion pour tout le lot, astuce de Montgomery). Retourne le nombre de clés.
// Une clé nulle (ou multiple de n) donne le point à l'infini (is_inf[j]).
__device__ __noinline__ unsigned int derive_thread_batch(
    BN256* out_x,
    BN256* out_y,
    bool* is_inf,
    const unsigned char* private_keys,
    const uint32_t* table,
    unsigned int num_keys,
    unsigned int window_bits,
    unsigned int first,
    bool endo
) {
    unsigned int count = num_keys - first;
    if (count > KEYS_PER_THREAD) count = KEYS_PER_THREAD;

    ECPointJ points[KEYS_PER_THREAD];
    FE prefix[KEYS_PER_THREAD];  // prefix[j] = Z0 * Z1 * ... * Zj

    for (unsigned int j = 0; j < count; j++) {
        BN256 scalar;
        bn_load_be(&scalar, &private_keys[(size_t)(first + j) * 32]);
        scalar_mult_base(&points[j], &scalar, table, window_bits);

        is_inf[j] = points[j].is_inf;
        if (points[j].is_inf) fe_set_one(&points[j].z);
        if (j == 0) prefix[0] = points[0].z;
        else fe_mul_call(&prefix[j], &prefix[j - 1], &points[j].z);
    }

    FE inv, beta;
    fe_inv(&inv, &prefix[count - 1]);
    fe_from_limbs(&beta, SECP256K1_BETA);

    for (int j = (int)count - 1; j >= 0; j--) {
        FE zinv, zinv2, ax, ay;
        if (j > 0) {
            fe_mul_call(&zinv, &inv, &prefix[j - 1]);
            fe_mul_call(&inv, &inv, &points[j].z);
        } else {
            zinv = inv;
        }

        fe_sqr_call(&zinv2, &zinv);
        fe_mul_call(&ax, &points[j].x, &zinv2);     // mag 6 * 1
        fe_mul_call(&zinv2, &zinv2, &zinv);
        fe_mul_call(&ay, &points[j].y, &zinv2);     // mag 3 * 1

        fe_to_bn_call(&out_x[3 * j], &ax);
        fe_to_bn_call(&out_y[j], &ay);

        if (endo) {
            fe_mul_call(&ax, &ax, &beta);           // beta * x
            fe_to_bn_call(&out_x[3 * j + 1], &ax);
            fe_mul_call(&ax, &ax, &beta);           // beta^2 * x
            fe_to_bn_call(&out_x[3 * j + 2], &ax);
        }
    }

    return count;
}

// Clés publiques complètes. Entrées/sorties : 32 octets big-endian par clé
// (format de la crate secp256k1). Point à l'infini -> sortie à zéro.
extern "C" __global__ void derive_public_keys_batch(
    const unsigned char* private_keys,
    const uint32_t* table,
    unsigned char* public_keys_x,
    unsigned char* public_keys_y,
    unsigned int num_keys,
    unsigned int window_bits
) {
    unsigned int first = (blockIdx.x * blockDim.x + threadIdx.x) * KEYS_PER_THREAD;
    if (first >= num_keys) return;

    BN256 ax[KEYS_PER_THREAD * 3], ay[KEYS_PER_THREAD];
    bool is_inf[KEYS_PER_THREAD];
    unsigned int count = derive_thread_batch(ax, ay, is_inf, private_keys, table, num_keys, window_bits, first, false);

    for (unsigned int j = 0; j < count; j++) {
        unsigned char* pub_x = &public_keys_x[(size_t)(first + j) * 32];
        unsigned char* pub_y = &public_keys_y[(size_t)(first + j) * 32];

        if (is_inf[j]) {
            for (int i = 0; i < 32; i++) { pub_x[i] = 0; pub_y[i] = 0; }
            continue;
        }

        bn_store_be(pub_x, &ax[3 * j]);
        bn_store_be(pub_y, &ay[j]);
    }
}

// HASH160 des clés publiques compressées : 20 octets par clé dérivée, soit 20 octets par
// clé privée sans endomorphisme, 6 x 20 avec (ordre des clés dérivées : voir plus haut).
// Point à l'infini -> sortie à zéro.
extern "C" __global__ void derive_hash160_batch(
    const unsigned char* private_keys,
    const uint32_t* table,
    unsigned char* hashes,
    unsigned int num_keys,
    unsigned int window_bits,
    unsigned int endo
) {
    unsigned int first = (blockIdx.x * blockDim.x + threadIdx.x) * KEYS_PER_THREAD;
    if (first >= num_keys) return;

    BN256 ax[KEYS_PER_THREAD * 3], ay[KEYS_PER_THREAD];
    bool is_inf[KEYS_PER_THREAD];
    unsigned int count = derive_thread_batch(ax, ay, is_inf, private_keys, table, num_keys, window_bits, first, endo != 0);

    const unsigned int stride = endo ? MAX_DERIVED : 1;
    for (unsigned int j = 0; j < count; j++) {
        unsigned char* out = &hashes[(size_t)(first + j) * stride * 20];

        if (is_inf[j]) {
            for (unsigned int i = 0; i < stride * 20; i++) out[i] = 0;
            continue;
        }

        uint32_t words[MAX_DERIVED][5];
        unsigned int derived = derived_hashes(words, &ax[3 * j], &ay[j], endo != 0);
        for (unsigned int i = 0; i < derived; i++) hash160_store(out + 20 * i, words[i]);
    }
}

// Filtre sur GPU : teste chaque HASH160 dérivé contre une liste d'intervalles
// [lo, hi] (10 mots big-endian par intervalle : lo puis hi, bornes incluses).
// Sortie : UN octet par clé privée, bit i = la clé dérivée i tombe dans un intervalle.
// Plus rien d'autre à rapatrier ; le CPU recalcule et vérifie les rares candidats.
extern "C" __global__ void derive_match_batch(
    const unsigned char* private_keys,
    const uint32_t* table,
    const uint32_t* ranges,
    unsigned int num_ranges,
    unsigned char* flags,
    unsigned int num_keys,
    unsigned int window_bits,
    unsigned int endo
) {
    unsigned int first = (blockIdx.x * blockDim.x + threadIdx.x) * KEYS_PER_THREAD;
    if (first >= num_keys) return;

    BN256 ax[KEYS_PER_THREAD * 3], ay[KEYS_PER_THREAD];
    bool is_inf[KEYS_PER_THREAD];
    unsigned int count = derive_thread_batch(ax, ay, is_inf, private_keys, table, num_keys, window_bits, first, endo != 0);

    for (unsigned int j = 0; j < count; j++) {
        unsigned char mask = 0;

        if (!is_inf[j]) {
            uint32_t words[MAX_DERIVED][5];
            unsigned int derived = derived_hashes(words, &ax[3 * j], &ay[j], endo != 0);
            for (unsigned int i = 0; i < derived; i++) {
                for (unsigned int r = 0; r < num_ranges; r++) {
                    if (hash160_in_range(words[i], ranges + 10 * r, ranges + 10 * r + 5)) {
                        mask |= (unsigned char)(1u << i);
                        break;
                    }
                }
            }
        }

        flags[first + j] = mask;
    }
}
