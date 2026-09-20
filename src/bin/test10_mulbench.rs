// Micro-benchmark des variantes de multiplication modulaire (kernels/bench_mul.cu).
//
// Compiler le kernel puis lancer :
//   nvcc -ptx -arch=sm_120 -O3 kernels/bench_mul.cu -o target/bench_mul.ptx
//   cargo run --bin test10_mulbench --release -- target/bench_mul.ptx

use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceBuffer;
use rustacuda::prelude::*;
use std::ffi::CString;
use std::time::Instant;

const NUM_THREADS: u32 = 1 << 16;
const ITERS: u32 = 4000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_path = std::env::args().nth(1).unwrap_or_else(|| "target/bench_mul.ptx".to_string());

    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _context = Context::create_and_push(ContextFlags::MAP_HOST, device)?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    let module = Module::load_from_file(&CString::new(ptx_path)?)?;

    let mut d_out = unsafe { DeviceBuffer::<u32>::uninitialized(NUM_THREADS as usize * 8)? };
    let mut reference: Option<Vec<u32>> = None;

    println!("{} threads x {} multiplications\n", NUM_THREADS, ITERS);

    for name in ["bench_mul_current", "bench_mul_comba", "bench_mul_fe26"] {
        let function = module.get_function(&CString::new(name)?)?;
        let mut best = f64::MAX;

        // 1 tour de chauffe + 3 mesures, on garde la meilleure
        for round in 0..4 {
            let start = Instant::now();
            unsafe {
                stream.launch(
                    &function,
                    GridSize::x(NUM_THREADS / 256),
                    BlockSize::x(256),
                    0,
                    &[
                        &mut d_out.as_device_ptr() as *mut _ as *mut std::ffi::c_void,
                        &mut { NUM_THREADS } as *mut _ as *mut std::ffi::c_void,
                        &mut { ITERS } as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
            stream.synchronize()?;
            if round > 0 {
                best = best.min(start.elapsed().as_secs_f64());
            }
        }

        let mut out = vec![0u32; NUM_THREADS as usize * 8];
        d_out.copy_to(&mut out[..])?;
        let same = match &reference {
            None => {
                reference = Some(out);
                "reference".to_string()
            }
            Some(r) => if *r == out { "identical ✓".to_string() } else { "DIFFERENT ✗".to_string() },
        };

        let mul_per_sec = NUM_THREADS as f64 * ITERS as f64 / best;
        println!("{:<20} {:>8.1} ms   {:>7.0} M mul/s   {}", name, best * 1000.0, mul_per_sec / 1e6, same);
    }

    Ok(())
}
