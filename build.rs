use std::path::{Path, PathBuf};
use std::process::Command;

/// Dernier sous-dossier (ordre alphabétique) de `dir`, ex: la version MSVC la plus récente
fn last_subdir(dir: &Path) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.pop()
}

fn find_nvcc() -> Option<PathBuf> {
    let cuda = match std::env::var_os("CUDA_PATH") {
        Some(path) => PathBuf::from(path),
        None => last_subdir(Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"))?,
    };
    let nvcc = cuda.join("bin").join("nvcc.exe");
    nvcc.exists().then_some(nvcc)
}

/// Dossier contenant cl.exe (requis par nvcc sous Windows)
fn find_msvc_bin() -> Option<PathBuf> {
    let vs = Path::new(r"C:\Program Files\Microsoft Visual Studio\2022");
    for edition in ["Community", "Professional", "Enterprise", "BuildTools"] {
        let msvc = vs.join(edition).join(r"VC\Tools\MSVC");
        if let Some(version) = last_subdir(&msvc) {
            let bin = version.join(r"bin\Hostx64\x64");
            if bin.join("cl.exe").exists() {
                return Some(bin);
            }
        }
    }
    None
}

/// Recompile le kernel. Retourne false si nvcc est introuvable ou échoue.
fn compile_kernel(cu_file: &Path, ptx_file: &Path, keys_per_thread: &str) -> bool {
    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found, using pre-compiled PTX");
        return false;
    };

    // RTX 5060 = compute capability 12.0
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_120".to_string());

    let mut cmd = Command::new(nvcc);
    cmd.arg("-ptx")
        .arg(format!("-arch={}", arch))
        .arg("-O3")
        .arg(format!("-DKEYS_PER_THREAD={}", keys_per_thread))
        .arg(cu_file)
        .arg("-o")
        .arg(ptx_file);
    if let Some(msvc_bin) = find_msvc_bin() {
        cmd.arg("-ccbin").arg(msvc_bin);
    }

    match cmd.status() {
        Ok(status) if status.success() => true,
        _ => {
            println!("cargo:warning=nvcc failed to compile {}", cu_file.display());
            false
        }
    }
}

fn main() {
    let cu_file = PathBuf::from("kernels/secp256k1_kernel.cu");
    let ptx_file = PathBuf::from("kernels/secp256k1_kernel.ptx");

    println!("cargo:rerun-if-changed={}", cu_file.display());
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=KEYS_PER_THREAD");

    // Clés par thread GPU : même valeur pour le kernel (-D) et pour src/gpu.rs (env!)
    let keys_per_thread = std::env::var("KEYS_PER_THREAD").unwrap_or_else(|_| "32".to_string());
    keys_per_thread.parse::<u32>().expect("KEYS_PER_THREAD must be a number");
    println!("cargo:rustc-env=GPU_KEYS_PER_THREAD={}", keys_per_thread);

    // Recompiler le PTX pour qu'il ne soit jamais en retard sur le .cu
    let compiled = compile_kernel(&cu_file, &ptx_file, &keys_per_thread);

    if !ptx_file.exists() {
        panic!("CUDA kernel PTX not found at {}. Please compile with:\n  nvcc -ptx -arch=sm_120 -O3 kernels/secp256k1_kernel.cu -o kernels/secp256k1_kernel.ptx", ptx_file.display());
    }
    if !compiled {
        println!("cargo:warning=Using pre-compiled (possibly stale) CUDA kernel: {}", ptx_file.display());
    }

    // Passer le chemin PTX au code Rust
    println!("cargo:rustc-env=CUDA_KERNEL_PTX={}", ptx_file.canonicalize().unwrap().display());
}
