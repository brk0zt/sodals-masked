use anyhow::Result;
use candle_core::quantized::gguf_file;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let path = if args.len() > 2 && args[1] == "--model" {
        args[2].clone()
    } else {
        std::env::current_dir()?.join("../../../models/Qwen3-8B-Q4_K_M.gguf").to_string_lossy().to_string()
    };
    
    println!("Reading: {}", path);
    let mut file = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut file)?;
    
    println!("Metadata:");
    for (k, v) in &content.metadata {
        if k.contains("count") || k.contains("size") || k.contains("length") || k.contains("head") {
            println!("  {}: {:?}", k, v);
        }
    }
    
    println!("Tensors (blk.0, blk.3, blk.31):");
    for layer in [0, 3, 31] {
        println!("  Layer {}:", layer);
        for k in content.tensor_infos.keys() {
            if k.contains(&format!("blk.{}", layer)) {
                let info = &content.tensor_infos[k];
                println!("    {}: shape={:?} type={:?}", k, info.shape.dims(), info.ggml_dtype);
            }
        }
    }
    Ok(())
}
