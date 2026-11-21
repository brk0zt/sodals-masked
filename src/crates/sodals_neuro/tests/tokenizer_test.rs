use sodals_neuro::llm_core::LlmEngine;
use tokenizers::Tokenizer;

#[test]
fn test_qwen_chatml_tokenization() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let tokenizer_path = manifest_dir.join("../../../models/tokenizer.json");
    let t = Tokenizer::from_file(&tokenizer_path)
        .or_else(|_| Tokenizer::from_file("c:/Users/Burak/Desktop/sodals_core/models/tokenizer.json"))
        .unwrap();
    let prompt = "<|im_start|>system\nYou are a Radiologist\n<|im_end|>";
    
    // Test without add_special_tokens=true (how generation_loop calls it for completely pre-formatted prompts)
    let encoded = t.encode(prompt, false).unwrap();
    let ids = encoded.get_ids();
    
    println!("Tokens: {:?}", ids);
    
    let im_start_id = t.token_to_id("<|im_start|>").unwrap_or(248045);
    assert!(ids.contains(&im_start_id) || ids.contains(&248045) || ids.contains(&151644), "Tokenizer DID NOT recognize <|im_start|> as a special token!");
}
