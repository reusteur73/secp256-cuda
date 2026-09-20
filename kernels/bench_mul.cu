// Micro-benchmark : variantes de la multiplication modulaire 256 bits sur GPU.
// Chaque thread enchaîne `iters` multiplications x = x * y ; les trois kernels doivent
// produire exactement le même résultat (8 limbs de 32 bits, forme canonique).
//
// Compilation (voir src/bin/test10_mulbench.rs) :
//   nvcc -ptx -arch=sm_120 -O3 kernels/bench_mul.cu -o target/bench_mul.ptx

#include "secp256k1_kernel.cu"

__device__ void bench_init(BN256* x, BN256* y, unsigned int idx) {
    for (int i = 0; i < 8; i++) {
        x->v[i] = 0x9E3779B9u * (idx + 1) + 0x7F4A7C15u * (i + 1);
        y->v[i] = 0x85EBCA6Bu * (i + 3) ^ (idx * 0xC2B2AE35u);
    }
    x->v[7] &= 0x7FFFFFFF;  // < p
    y->v[7] &= 0x7FFFFFFF;
}

// ---- Variante A : bn_mul actuel (8x32, balayage par opérande) ----

extern "C" __global__ void bench_mul_current(uint32_t* out, unsigned int n, unsigned int iters) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    BN256 x, y;
    bench_init(&x, &y, idx);
    for (unsigned int i = 0; i < iters; i++) bn_mul(&x, &x, &y);
    for (int i = 0; i < 8; i++) out[idx * 8 + i] = x.v[i];
}

// ---- Variante B : 8x32, accumulation par colonnes (Comba) ----

__device__ void bn_mul_comba(BN256* dst, const BN256* a, const BN256* b) {
    uint32_t t[16];
    uint64_t acc = 0;   // 64 bits bas de l'accumulateur
    uint32_t acc_hi = 0; // débordements au-delà de 64 bits

    for (int k = 0; k < 15; k++) {
        int lo = k < 8 ? 0 : k - 7;
        int hi = k < 8 ? k : 7;
        for (int i = lo; i <= hi; i++) {
            uint64_t p = (uint64_t)a->v[i] * b->v[k - i];
            acc += p;
            acc_hi += (acc < p);
        }
        t[k] = (uint32_t)acc;
        acc = (acc >> 32) | ((uint64_t)acc_hi << 32);
        acc_hi = 0;
    }
    t[15] = (uint32_t)acc;

    bn_reduce(dst, t);
}

extern "C" __global__ void bench_mul_comba(uint32_t* out, unsigned int n, unsigned int iters) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    BN256 x, y;
    bench_init(&x, &y, idx);
    for (unsigned int i = 0; i < iters; i++) bn_mul_comba(&x, &x, &y);
    for (int i = 0; i < 8; i++) out[idx * 8 + i] = x.v[i];
}

// ---- Variante C : 10 limbs de 26 bits (représentation de libsecp256k1 en 32 bits) ----
// Les produits de colonnes tiennent dans 64 bits sans propagation de retenue.

typedef struct {
    uint32_t n[10];
} FE26;

#define M26 0x3FFFFFFu

__device__ void fe26_from_bn(FE26* r, const BN256* a) {
    for (int i = 0; i < 10; i++) {
        int pos = 26 * i;
        int limb = pos >> 5, sh = pos & 31;
        uint64_t v = a->v[limb] >> sh;
        if (limb < 7) v |= (uint64_t)a->v[limb + 1] << (32 - sh);
        r->n[i] = (uint32_t)v & M26;
    }
}

// Résultat faiblement normalisé : limbs <= 26 bits (+1), valeur < 2^257 environ
__device__ void fe26_mul(FE26* r, const FE26* a, const FE26* b) {
    uint64_t c[19];
    for (int k = 0; k < 19; k++) c[k] = 0;
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            c[i + j] += (uint64_t)a->n[i] * b->n[j];

    // Partie haute (colonnes 10..18) en limbs de 26 bits ; h[9] garde le reste
    uint64_t h[10];
    uint64_t carry = 0;
    for (int k = 0; k < 9; k++) {
        carry += c[10 + k];
        h[k] = carry & M26;
        carry >>= 26;
    }
    h[9] = carry;

    // 2^260 = 2^36 + 0x3D10 (mod p)  =>  colonne k+10 -> 0x3D10 en k, 0x400 en k+1
    uint64_t d[11];
    for (int k = 0; k < 10; k++) d[k] = c[k] + h[k] * 0x3D10u + (k > 0 ? h[k - 1] * 0x400u : 0);
    d[10] = h[9] * 0x400u;

    carry = 0;
    for (int k = 0; k < 9; k++) {
        carry += d[k];
        r->n[k] = (uint32_t)carry & M26;
        carry >>= 26;
    }
    carry += d[9];
    r->n[9] = (uint32_t)carry & 0x3FFFFFu;  // 22 bits : 9*26 + 22 = 256
    uint64_t top = (carry >> 22) + (d[10] << 4);  // en unités de 2^256

    // 2^256 = 2^32 + 0x3D1 (mod p)
    carry = (uint64_t)r->n[0] + top * 0x3D1u;
    r->n[0] = (uint32_t)carry & M26;
    carry >>= 26;
    carry += (uint64_t)r->n[1] + (top << 6);
    r->n[1] = (uint32_t)carry & M26;
    carry >>= 26;
    for (int k = 2; k < 9; k++) {
        carry += r->n[k];
        r->n[k] = (uint32_t)carry & M26;
        carry >>= 26;
    }
    r->n[9] += (uint32_t)carry;
}

// Forme canonique 8x32 (réduction complète)
__device__ void fe26_to_bn(BN256* r, const FE26* a) {
    // Recomposer en 8x32 + dépassement, puis réutiliser la réduction existante
    uint32_t t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;
    uint64_t acc = 0;
    int bits = 0, out = 0;
    for (int i = 0; i < 10; i++) {
        acc |= (uint64_t)a->n[i] << bits;
        bits += (i < 9) ? 26 : 32;
        while (bits >= 32) {
            t[out++] = (uint32_t)acc;
            acc >>= 32;
            bits -= 32;
        }
    }
    if (out < 16) t[out] = (uint32_t)acc;
    bn_reduce(r, t);
}

extern "C" __global__ void bench_mul_fe26(uint32_t* out, unsigned int n, unsigned int iters) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    BN256 x, y;
    bench_init(&x, &y, idx);
    FE26 fx, fy;
    fe26_from_bn(&fx, &x);
    fe26_from_bn(&fy, &y);
    for (unsigned int i = 0; i < iters; i++) fe26_mul(&fx, &fx, &fy);
    fe26_to_bn(&x, &fx);
    for (int i = 0; i < 8; i++) out[idx * 8 + i] = x.v[i];
}
