use candle_core::quantized::gguf_file;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open("models/Qwen3.5-9B_Q4_K_M.gguf")?;
    let gguf = gguf_file::Content::read(&mut file)?;
    
    // Print Q/K norm shapes for ALL layers
    println!("=== Q/K Norm Weight Shapes (ALL layers) ===");
    for (name, info) in gguf.tensor_infos.iter() {
        if name.contains("attn_q_norm") || name.contains("attn_k_norm") {
            println!("  {}: {:?}", name, info.shape);
        }
    }
    
    // Print ssm_out weight shapes
    println!("\n=== SSM Output Weight Shapes ===");
    for (name, info) in gguf.tensor_infos.iter() {
        if name.contains("ssm_out.weight") {
            println!("  {}: {:?}", name, info.shape);
        }
    }
    
    // Print ffn_down weight shapes 
    println!("\n=== FFN Down Weight Shapes (first 3) ===");
    for (name, info) in gguf.tensor_infos.iter() {
        if name.contains("ffn_down.weight") {
            println!("  {}: {:?}", name, info.shape);
        }
    }
    
    Ok(())
}
