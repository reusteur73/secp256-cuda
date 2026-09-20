# secp256-cuda

GPU-accelerated Bitcoin P2PKH address generation, written in Rust with CUDA kernels.

Random private keys go in; the GPU computes the secp256k1 public key `k*G` and its
HASH160 (`RIPEMD160(SHA256(compressed pubkey))`), and only the keys matching a condition
come back. The bundled condition is a **vanity address** search: find a key whose address
starts with a chosen prefix such as `1GPU`.

Measured on a GTX 1080 Ti: **54.8 M keys/s** (`endo` mode).

Measured on a RTX 5060 Portable: **198 M keys/s** (`endo` mode).

## How it works

```
[RNG thread]   random keys prepared ahead of time
[GPU thread]   keys -> pinned memory -> k*G + HASH160 on GPU (2 batches in flight)
[main/Rayon]   condition check -> address + private key for the hits
```

- **Fixed-base windowed multiplication**: a precomputed table of multiples of `G` with
  signed digits turns `k*G` into ~12 point additions (22-bit windows, 1.6 GB table,
  cached on disk in `target/gpu_tables/`).
- **GLV endomorphism + negation**: one `k*G` yields 6 public keys
  (`±k`, `±λk`, `±λ²k`), so 6 addresses for the price of one scalar multiplication.
- **On-GPU pre-filter**: the prefix is converted to HASH160 ranges; the GPU returns one
  flag byte per key instead of 20-byte hashes, and the CPU re-checks only the candidates.
- **Async pipeline**: CUDA streams and pinned host memory overlap transfers and compute.
- **One kernel per GPU architecture**: `build.rs` compiles a PTX per compute capability
  and the right one is picked at runtime.

## Requirements

- Rust (stable)
- CUDA toolkit with `nvcc`: Pascal (`sm_61`) needs CUDA ≤ 12.9, Blackwell (`sm_120`) needs CUDA ≥ 12.8
- A host compiler for `nvcc`: MSVC 2022 on Windows, `g++` ≤ 14 on Linux
- An NVIDIA GPU

## Usage

```sh
# Vanity search: [num_keys] [batch_size] [prefix] [mode]
cargo run --release --bin test9_condition -- 64000000 1000000 1GPU endo

# Full pipeline benchmark, hashing on CPU: [num_keys] [batch_size]
cargo run --release --bin test8_pipeline -- 16000000 1000000
```

Modes of `test9_condition`:

| Mode     | What the GPU returns                  | Keys per `k*G` |
|----------|---------------------------------------|----------------|
| `cpu`    | 20-byte HASH160, condition on CPU     | 1              |
| `single` | 1 flag byte, pre-filtered on GPU      | 1              |
| `endo`   | 1 flag byte, pre-filtered on GPU      | 6              |

The other `test*` binaries validate the GPU arithmetic against the `secp256k1` crate and
benchmark individual stages.

## Configuration

| Variable          | When    | Default       | Effect                                                   |
|-------------------|---------|---------------|----------------------------------------------------------|
| `CUDA_ARCH`       | build   | detected GPUs | Architectures to compile, e.g. `sm_61,sm_120`            |
| `CUDA_PATH`       | build   | auto          | CUDA toolkit location                                    |
| `KEYS_PER_THREAD` | build   | `32`          | Keys processed by each GPU thread                        |
| `GPU_WINDOW_BITS` | runtime | `22`          | Table window width (4–24): 16 → 36 MB, 22 → 1.6 GB       |
| `GPU_BLOCK_SIZE`  | runtime | `32` / `128`  | CUDA threads per block (32 on Pascal and older)          |

## Layout

```
kernels/   CUDA sources (secp256k1 field/point arithmetic, SHA-256, RIPEMD-160)
src/gpu.rs       GPU context, precomputed table, async batch slots
src/endo.rs      private keys of the 6 endomorphism-derived points
src/address.rs   Base58Check, prefix -> HASH160 ranges
src/bin/         validation tests and benchmarks
```

## Disclaimer

For education and vanity address generation. Searching for keys of existing addresses is
hopeless by design: the HASH160 space has 2^160 elements.
