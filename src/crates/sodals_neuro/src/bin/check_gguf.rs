use candle_core::quantized::gguf_file;
use std::collections::HashMap;

fn main() {
    let path = r"C:\Users\Burak\Desktop\sodals_core\models\Qwen3.5-9B_Q4_K_M.gguf";
    let mut file = std::fs::File::open(path).unwrap();
    let content = gguf_file::Content::read(&mut file).unwrap();
    let metadata: &HashMap<String, gguf_file::Value> = &content.metadata;

    // Print specific string values
    let string_keys = [
        "general.architecture",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.chat_template",
    ];
    for key in &string_keys {
        if let Some(val) = metadata.get(*key) {
            if let Ok(s) = val.to_string() {
                println!("{} = {}", key, s);
            }
        }
    }

    // Print all bool/uint32/f32 metadata
    println!("\n=== All scalar metadata ===");
    for (key, val) in metadata.iter() {
        if let Ok(v) = val.to_u32() {
            if !key.contains("tokenizer.ggml.tokens") && !key.contains("tokenizer.ggml.merges") && !key.contains("tokenizer.ggml.token_type") {
                println!("  {} = {} (u32)", key, v);
            }
        } else if let Ok(v) = val.to_f32() {
            println!("  {} = {} (f32)", key, v);
        } else if let Ok(v) = val.to_bool() {
            println!("  {} = {} (bool)", key, v);
        }
    }
}
