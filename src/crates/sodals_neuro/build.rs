use std::env;
use std::fs;
use std::path::PathBuf;


fn main() {
    // DeIIiIIiklikleri izle
    println!("cargo:rerun-if-changed=src/kernels/physics_mask.cu");
    println!("cargo:rerun-if-changed=src/kernels/cublas_cgemm.cu");
    println!("cargo:rerun-if-changed=src/kernels/fluid_algebra.cu");
    println!("cargo:rerun-if-changed=build.rs");

    compile_cuda_kernels();
}

fn compile_cuda_kernels() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // --- DIIZELTME: Mutlak Yol KullanIImII ---
    // CARGO_MANIFEST_DIR = build.rs'in bulunduIIu klasIIr (sodals_neuro)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_source = manifest_dir.join("src/kernels/physics_mask.cu");
    
    let _ptx_output = out_dir.join("physics_mask.ptx");
    let target_ptx = PathBuf::from("target/physics_mask.ptx");

    // 1. DOSYA VARLIK KONTROLII (Debug IIIIin Kritik)
    if !kernel_source.exists() {
        // Dosya yoksa panik yap ve tam yolu gIIster
        panic!(
            "\n\nIIII KRIITIIK HATA: CUDA Kaynak dosyasII bulunamadII!\n\
            III Aranan Yol: {:?}\n\
            IIII IIIIzIIm: 'physics_mask.cu' dosyasIInIIn 'sodals_neuro/src/kernels/' iIIinde olduIIundan emin olun.\n\n",
            kernel_source
        );
    }

    // Hedef klasIIr oluIItur
    if let Some(parent) = target_ptx.parent() {
        let _ = fs::create_dir_all(parent);
    }

    // CUDA Yolunu Bul
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    let nvcc_name = if cfg!(target_os = "windows") { "bin/nvcc.exe" } else { "bin/nvcc" };
    let nvcc_path = PathBuf::from(&cuda_path).join(nvcc_name);
    
    // 3. Compile CUDA Kernel via Command (Bypassing cc-rs nvcc macro detection)
    println!("cargo:warning=Compiling CUDA kernel (static library): {:?}", kernel_source);
    
    let target_obj = out_dir.join(if cfg!(target_os = "windows") { "physics_mask.obj" } else { "physics_mask.o" });
    
    let mut command = std::process::Command::new(&nvcc_path);
    command.arg("-c")
        .arg(&kernel_source)
        .arg("-o").arg(&target_obj)
        .arg("-m64")
        .arg("-arch=sm_75")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("-lineinfo")
        .arg("-allow-unsupported-compiler")
        .arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
        
    if cfg!(target_os = "windows") {
        let host_compiler = cc::Build::new().get_compiler();
        command.arg(format!("-ccbin={}", host_compiler.path().display()));
        command.arg("-Xcompiler").arg("/MT");
    }
    
    let status = command.status().expect("Failed to execute nvcc manually");
    assert!(status.success(), "NVCC MSVC/Windows binding failed. Verify your CUDA toolkit.");

    // Link into static library using cc-rs (handles MSVC lib.exe binding silently)
    cc::Build::new()
        .cargo_metadata(false)
        .object(&target_obj)
        .compile("physics_mask");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=physics_mask");

    // PHASE 17: Compile Fused Resonance Core kernel
    let kernel_fused = manifest_dir.join("src/kernels/fused_resonance_core.cu");
    if kernel_fused.exists() {
        println!("cargo:rerun-if-changed=src/kernels/fused_resonance_core.cu");
        let target_fused_obj = out_dir.join(if cfg!(target_os = "windows") { "fused_resonance_core.obj" } else { "fused_resonance_core.o" });
        println!("cargo:warning=Compiling CUDA kernel (fused resonance): {:?}", kernel_fused);
        
        let mut cmd_spec = std::process::Command::new(&nvcc_path);
        cmd_spec.arg("-c").arg(&kernel_fused).arg("-o").arg(&target_fused_obj)
            .arg("-m64").arg("-arch=sm_75").arg("-O3").arg("--use_fast_math")
            .arg("-lineinfo").arg("-allow-unsupported-compiler").arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
            
        if cfg!(target_os = "windows") {
            let host_compiler = cc::Build::new().get_compiler();
            cmd_spec.arg(format!("-ccbin={}", host_compiler.path().display()));
            cmd_spec.arg("-Xcompiler").arg("/MT");
        }
        
        let status_spec = cmd_spec.status().expect("Failed to execute nvcc manually for fused_resonance_core");
        assert!(status_spec.success(), "NVCC MSVC/Windows binding failed for fused_resonance_core.");

        cc::Build::new().cargo_metadata(false).object(&target_fused_obj).compile("fused_resonance_core");
        println!("cargo:rustc-link-lib=static=fused_resonance_core");
    }

    // PHASE 18: Compile Atan2 Kernel
    let kernel_atan2 = manifest_dir.join("src/kernels/atan2.cu");
    if kernel_atan2.exists() {
        println!("cargo:rerun-if-changed=src/kernels/atan2.cu");
        let target_atan2_obj = out_dir.join(if cfg!(target_os = "windows") { "atan2_kernel.obj" } else { "atan2_kernel.o" });
        println!("cargo:warning=Compiling CUDA kernel (atan2): {:?}", kernel_atan2);
        
        let mut cmd_spec = std::process::Command::new(&nvcc_path);
        cmd_spec.arg("-c").arg(&kernel_atan2).arg("-o").arg(&target_atan2_obj)
            .arg("-m64").arg("-arch=sm_75").arg("-O3").arg("--use_fast_math")
            .arg("-lineinfo").arg("-allow-unsupported-compiler").arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
            
        if cfg!(target_os = "windows") {
            let host_compiler = cc::Build::new().get_compiler();
            cmd_spec.arg(format!("-ccbin={}", host_compiler.path().display()));
            cmd_spec.arg("-Xcompiler").arg("/MT");
        }
        
        let status_spec = cmd_spec.status().expect("Failed to execute nvcc manually for atan2");
        assert!(status_spec.success(), "NVCC MSVC/Windows binding failed for atan2.");

        cc::Build::new().cargo_metadata(false).object(&target_atan2_obj).compile("atan2_kernel");
        println!("cargo:rustc-link-lib=static=atan2_kernel");
    }

    // PHASE 19: Compile GNN Message Pass Kernel
    let kernel_gnn = manifest_dir.join("src/kernels/gnn_message_pass.cu");
    if kernel_gnn.exists() {
        println!("cargo:rerun-if-changed=src/kernels/gnn_message_pass.cu");
        let target_gnn_obj = out_dir.join(if cfg!(target_os = "windows") { "gnn_message_pass.obj" } else { "gnn_message_pass.o" });
        println!("cargo:warning=Compiling CUDA kernel (gnn message pass): {:?}", kernel_gnn);
        
        let mut cmd_spec = std::process::Command::new(&nvcc_path);
        cmd_spec.arg("-c").arg(&kernel_gnn).arg("-o").arg(&target_gnn_obj)
            .arg("-m64").arg("-arch=sm_75").arg("-O3").arg("--use_fast_math")
            .arg("-lineinfo").arg("-allow-unsupported-compiler").arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
            
        if cfg!(target_os = "windows") {
            let host_compiler = cc::Build::new().get_compiler();
            cmd_spec.arg(format!("-ccbin={}", host_compiler.path().display()));
            cmd_spec.arg("-Xcompiler").arg("/MT");
        }
        
        let status_spec = cmd_spec.status().expect("Failed to execute nvcc manually for gnn_message_pass");
        assert!(status_spec.success(), "NVCC MSVC/Windows binding failed for gnn_message_pass.");

        cc::Build::new().cargo_metadata(false).object(&target_gnn_obj).compile("gnn_message_pass");
        println!("cargo:rustc-link-lib=static=gnn_message_pass");
    }

    // PHASE 20: Compile cuBLAS Complex GEMM Kernel
    let kernel_cgemm = manifest_dir.join("src/kernels/cublas_cgemm.cu");
    if kernel_cgemm.exists() {
        let target_cgemm_obj = out_dir.join(if cfg!(target_os = "windows") { "cublas_cgemm.obj" } else { "cublas_cgemm.o" });
        println!("cargo:warning=Compiling CUDA kernel (cublas cgemm): {:?}", kernel_cgemm);
        
        let mut cmd_spec = std::process::Command::new(&nvcc_path);
        cmd_spec.arg("-c").arg(&kernel_cgemm).arg("-o").arg(&target_cgemm_obj)
            .arg("-m64").arg("-arch=sm_75").arg("-O3").arg("--use_fast_math")
            .arg("-lineinfo").arg("-allow-unsupported-compiler").arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
            
        if cfg!(target_os = "windows") {
            let host_compiler = cc::Build::new().get_compiler();
            cmd_spec.arg(format!("-ccbin={}", host_compiler.path().display()));
            cmd_spec.arg("-Xcompiler").arg("/MT");
        }
        
        let status_spec = cmd_spec.status().expect("Failed to execute nvcc manually for cublas_cgemm");
        assert!(status_spec.success(), "NVCC MSVC/Windows binding failed for cublas_cgemm.");

        cc::Build::new().cargo_metadata(false).object(&target_cgemm_obj).compile("cublas_cgemm");
        println!("cargo:rustc-link-lib=static=cublas_cgemm");
    }

    // PHASE 21: Compile Fluid Algebra Kernel
    let kernel_fluid = manifest_dir.join("src/kernels/fluid_algebra.cu");
    if kernel_fluid.exists() {
        let target_fluid_obj = out_dir.join(if cfg!(target_os = "windows") { "fluid_algebra.obj" } else { "fluid_algebra.o" });
        println!("cargo:warning=Compiling CUDA kernel (fluid algebra): {:?}", kernel_fluid);
        
        let mut cmd_spec = std::process::Command::new(&nvcc_path);
        cmd_spec.arg("-c").arg(&kernel_fluid).arg("-o").arg(&target_fluid_obj)
            .arg("-m64").arg("-arch=sm_75").arg("-O3").arg("--use_fast_math")
            .arg("-lineinfo").arg("-allow-unsupported-compiler").arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH");
            
        if cfg!(target_os = "windows") {
            let host_compiler = cc::Build::new().get_compiler();
            cmd_spec.arg(format!("-ccbin={}", host_compiler.path().display()));
            cmd_spec.arg("-Xcompiler").arg("/MT");
        }
        
        let status_spec = cmd_spec.status().expect("Failed to execute nvcc manually for fluid_algebra");
        assert!(status_spec.success(), "NVCC MSVC/Windows binding failed for fluid_algebra.");

        cc::Build::new().cargo_metadata(false).object(&target_fluid_obj).compile("fluid_algebra");
        println!("cargo:rustc-link-lib=static=fluid_algebra");
    }

    // Link cuBLAS
    println!("cargo:rustc-link-lib=cublas");

    // --- HOST-SIDE MEMORY LIBRARY DERLEMESI ---
    let host_memory_source = manifest_dir.join("src/kernels/host_memory.cpp");
    let cuda_include = PathBuf::from(&cuda_path).join("include");
    
    // KIItIIphane YollarIInII Ekle
    let cuda_lib64 = PathBuf::from(&cuda_path).join("lib/x64");
    println!("cargo:rustc-link-search=native={}", cuda_lib64.display());

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}", manifest_dir);
    
    let cuda_lib = PathBuf::from(&cuda_path).join("lib");
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());

    // Host-side c++ codunu C++ Runtime ile derle
    cc::Build::new()
        .cpp(true)
        .file(&host_memory_source)
        .include(&cuda_include)
        .flag_if_supported("-std=c++14")
        .compile("host_memory");
        
    println!("cargo:rerun-if-changed={}", host_memory_source.display());

    // Link cudart for host memory functions
    println!("cargo:rustc-link-lib=cudart");
}













