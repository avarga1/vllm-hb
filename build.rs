use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/rms_norm.cu");
    println!("cargo:rerun-if-changed=kernels/rope.cu");

    // Only compile custom CUDA kernels when the `cuda` feature is active.
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_owned());

    let nvcc = PathBuf::from(&cuda_home).join("bin").join("nvcc");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Compile to PTX (virtual ISA).  The CUDA driver JIT-compiles PTX to SASS
    // at load time, so one virtual arch covers all GPUs from that SM onward.
    // We target compute_70 (Volta) as the baseline — runs on V100, A100, H100.
    let nvcc_ptx_args = |cu: &str, ptx: &PathBuf| {
        vec![
            "-ptx".to_owned(),
            "-O3".to_owned(),
            "-arch=compute_70".to_owned(),
            "-o".to_owned(),
            ptx.to_str().unwrap().to_owned(),
            format!("kernels/{cu}"),
        ]
    };

    // ── rms_norm ─────────────────────────────────────────────────────────────
    let rms_ptx = out_dir.join("rms_norm.ptx");
    let status = std::process::Command::new(&nvcc)
        .args(nvcc_ptx_args("rms_norm.cu", &rms_ptx))
        .status()
        .expect("nvcc not found — set CUDA_HOME");
    assert!(status.success(), "nvcc failed compiling rms_norm.cu");

    // ── rope ─────────────────────────────────────────────────────────────────
    let rope_ptx = out_dir.join("rope.ptx");
    let status = std::process::Command::new(&nvcc)
        .args(nvcc_ptx_args("rope.cu", &rope_ptx))
        .status()
        .expect("nvcc not found — set CUDA_HOME");
    assert!(status.success(), "nvcc failed compiling rope.cu");

    // Expose OUT_DIR path so the Rust code can include_str! the .ptx files.
    println!("cargo:rustc-env=KERNEL_OUT_DIR={}", out_dir.display());
}
