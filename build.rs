use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/rms_norm.cu");
    println!("cargo:rerun-if-changed=kernels/rope.cu");

    // Only compile custom CUDA kernels when the `cuda` feature is active.
    // Without CUDA the kernels directory is ignored entirely.
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_owned());

    let nvcc = PathBuf::from(&cuda_home).join("bin").join("nvcc");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // ── rms_norm ─────────────────────────────────────────────────────────────
    let rms_ptx = out_dir.join("rms_norm.ptx");
    let status = std::process::Command::new(&nvcc)
        .args([
            "-ptx",
            "-O3",
            "--generate-code",
            "arch=compute_70,code=sm_70",   // V100
            "--generate-code",
            "arch=compute_80,code=sm_80",   // A100
            "--generate-code",
            "arch=compute_86,code=sm_86",   // A40 / RTX 3090
            "--generate-code",
            "arch=compute_89,code=sm_89",   // H100 (sm_89 compat)
            "-o",
            rms_ptx.to_str().unwrap(),
            "kernels/rms_norm.cu",
        ])
        .status()
        .expect("nvcc not found — set CUDA_HOME");

    assert!(status.success(), "nvcc failed compiling rms_norm.cu");

    // ── rope ─────────────────────────────────────────────────────────────────
    let rope_ptx = out_dir.join("rope.ptx");
    let status = std::process::Command::new(&nvcc)
        .args([
            "-ptx",
            "-O3",
            "--generate-code",
            "arch=compute_70,code=sm_70",
            "--generate-code",
            "arch=compute_80,code=sm_80",
            "--generate-code",
            "arch=compute_86,code=sm_86",
            "--generate-code",
            "arch=compute_89,code=sm_89",
            "-o",
            rope_ptx.to_str().unwrap(),
            "kernels/rope.cu",
        ])
        .status()
        .expect("nvcc not found — set CUDA_HOME");

    assert!(status.success(), "nvcc failed compiling rope.cu");

    // Expose OUT_DIR path so the Rust code can include_str! the .ptx files.
    println!("cargo:rustc-env=KERNEL_OUT_DIR={}", out_dir.display());
}
