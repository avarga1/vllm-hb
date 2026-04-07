fn main() {
    // candle handles CUDA linking internally via cudarc.
    // We keep this build.rs minimal — no manual libtorch linking needed.
    println!("cargo:rerun-if-changed=build.rs");
}
