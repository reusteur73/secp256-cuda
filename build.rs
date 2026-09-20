use std::path::{Path, PathBuf};
use std::process::Command;

const KERNEL_DIR: &str = "kernels";
const KERNEL_NAME: &str = "secp256k1_kernel";

/// Architectures compilées quand ni CUDA_ARCH ni nvidia-smi ne renseignent :
/// GTX 1080 Ti = compute capability 6.1, RTX 50XX = 12.0.
/// sm_61 demande CUDA <= 12.9 (CUDA 13 ne cible plus Pascal), sm_120 demande CUDA >= 12.8.
const DEFAULT_ARCHS: [&str; 2] = ["sm_61", "sm_120"];

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
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };

    let mut roots = Vec::new();
    if let Some(path) = std::env::var_os("CUDA_PATH") {
        roots.push(PathBuf::from(path));
    } else if cfg!(windows) {
        roots.extend(last_subdir(Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")));
    } else {
        // /opt/cuda : paquet Arch Linux
        roots.push(PathBuf::from("/opt/cuda"));
        roots.push(PathBuf::from("/usr/local/cuda"));
    }

    roots
        .iter()
        .map(|root| root.join("bin").join(exe))
        .find(|nvcc| nvcc.exists())
        // Sinon, nvcc du PATH
        .or_else(|| Command::new(exe).arg("--version").output().ok().map(|_| PathBuf::from(exe)))
}

/// Compilateur hôte à passer à nvcc (-ccbin), si nvcc ne peut pas le trouver seul
fn find_host_compiler() -> Option<PathBuf> {
    if cfg!(windows) {
        // Dossier contenant cl.exe
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
        return None;
    }

    // nvcc lit NVCC_CCBIN lui-même (défini par le paquet cuda d'Arch Linux)
    if std::env::var_os("NVCC_CCBIN").is_some() {
        return None;
    }
    // CUDA 12.9 refuse gcc > 14 : préférer un gcc versionné plus ancien s'il est installé
    ["g++-14", "g++-13", "g++-12"]
        .iter()
        .map(|gxx| Path::new("/usr/bin").join(gxx))
        .find(|gxx| gxx.exists())
}

/// Architectures des GPU de la machine, ex: compute capability "6.1" -> "sm_61"
fn detect_gpu_archs() -> Vec<String> {
    let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    else {
        return Vec::new();
    };

    let mut archs: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| line.trim().replace('.', ""))
        .filter(|cc| !cc.is_empty() && cc.chars().all(|c| c.is_ascii_digit()))
        .map(|cc| format!("sm_{}", cc))
        .collect();
    archs.sort();
    archs.dedup();
    archs
}

/// Architectures à compiler : CUDA_ARCH (ex: "sm_61" ou "sm_61,sm_120"), sinon celles des
/// GPU de la machine, sinon DEFAULT_ARCHS.
fn target_archs() -> Vec<String> {
    if let Ok(list) = std::env::var("CUDA_ARCH") {
        return list.split(',').map(|a| a.trim().to_string()).filter(|a| !a.is_empty()).collect();
    }
    let detected = detect_gpu_archs();
    if !detected.is_empty() {
        return detected;
    }
    DEFAULT_ARCHS.iter().map(|a| a.to_string()).collect()
}

fn ptx_path(arch: &str) -> PathBuf {
    Path::new(KERNEL_DIR).join(format!("{}.{}.ptx", KERNEL_NAME, arch))
}

/// Recompile le kernel pour `arch`. Retourne false si nvcc échoue.
fn compile_kernel(nvcc: &Path, cu_file: &Path, arch: &str, keys_per_thread: &str) -> bool {
    let mut cmd = Command::new(nvcc);
    cmd.arg("-ptx")
        .arg(format!("-arch={}", arch))
        .arg("-O3")
        .arg(format!("-DKEYS_PER_THREAD={}", keys_per_thread))
        .arg(cu_file)
        .arg("-o")
        .arg(ptx_path(arch));
    if let Some(ccbin) = find_host_compiler() {
        cmd.arg("-ccbin").arg(ccbin);
    }

    match cmd.status() {
        Ok(status) if status.success() => true,
        _ => {
            println!("cargo:warning=nvcc failed to compile {} for {}", cu_file.display(), arch);
            false
        }
    }
}

/// Compute capabilities (61, 120, ...) des PTX présents dans kernels/, fraîchement
/// compilés ou non
fn available_archs() -> Vec<u32> {
    let prefix = format!("{}.sm_", KERNEL_NAME);
    let mut archs: Vec<u32> = std::fs::read_dir(KERNEL_DIR)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|name| name.strip_prefix(&prefix)?.strip_suffix(".ptx")?.parse().ok())
        .collect();
    archs.sort();
    archs
}

fn main() {
    let cu_file = Path::new(KERNEL_DIR).join(format!("{}.cu", KERNEL_NAME));

    println!("cargo:rerun-if-changed={}", cu_file.display());
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=KEYS_PER_THREAD");

    // Clés par thread GPU : même valeur pour le kernel (-D) et pour src/gpu.rs (env!)
    let keys_per_thread = std::env::var("KEYS_PER_THREAD").unwrap_or_else(|_| "32".to_string());
    keys_per_thread.parse::<u32>().expect("KEYS_PER_THREAD must be a number");
    println!("cargo:rustc-env=GPU_KEYS_PER_THREAD={}", keys_per_thread);

    // Recompiler les PTX pour qu'ils ne soient jamais en retard sur le .cu
    let nvcc = find_nvcc();
    if nvcc.is_none() {
        println!("cargo:warning=nvcc not found, using pre-compiled PTX");
    }
    for arch in target_archs() {
        let compiled = nvcc.as_deref().is_some_and(|nvcc| compile_kernel(nvcc, &cu_file, &arch, &keys_per_thread));
        if !compiled && ptx_path(&arch).exists() {
            println!("cargo:warning=Using pre-compiled (possibly stale) CUDA kernel: {}", ptx_path(&arch).display());
        }
    }

    let archs = available_archs();
    if archs.is_empty() {
        panic!("No CUDA kernel PTX found in {}/. Please compile with:\n  nvcc -ptx -arch=sm_61 -O3 {} -o {}", KERNEL_DIR, cu_file.display(), ptx_path("sm_61").display());
    }

    // Passer les PTX au code Rust, qui choisit selon le GPU (voir select_ptx dans src/gpu.rs)
    let archs: Vec<String> = archs.iter().map(|a| a.to_string()).collect();
    println!("cargo:rustc-env=CUDA_KERNEL_ARCHS={}", archs.join(","));
    println!("cargo:rustc-env=CUDA_KERNEL_DIR={}", Path::new(KERNEL_DIR).canonicalize().unwrap().display());
}
