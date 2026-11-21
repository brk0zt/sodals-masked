use tokenizers::Tokenizer;

fn main() {
    let path = r"C:\Users\Burak\Desktop\sodals_core\models\tokenizer.json";
    let tok = Tokenizer::from_file(path).unwrap();

    let vocab_size_false = tok.get_vocab_size(false);
    let vocab_size_true = tok.get_vocab_size(true);
    println!("Vocab size (without added tokens): {}", vocab_size_false);
    println!("Vocab size (with added tokens):    {}", vocab_size_true);
    println!("  -> uses Qwen2 IDs (151k) = {}", vocab_size_true <= 200000);
    println!("  -> uses Qwen3 IDs (248k) = {}", vocab_size_true > 200000);

    let special_tokens = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|reserved_special_token_0|>",
    ];

    println!("\n--- Special token IDs from tokenizer ---");
    for token_str in &special_tokens {
        match tok.token_to_id(token_str) {
            Some(id) => println!("  {:<30} -> id={}", token_str, id),
            None => println!("  {:<30} -> NOT FOUND in vocab", token_str),
        }
    }

    println!("\n--- Encode special tokens directly ---");
    for token_str in &special_tokens {
        let enc = tok.encode(*token_str, false).unwrap();
        let ids = enc.get_ids();
        println!("  encode(\"{}\", false) = {:?}", token_str, ids);
    }

    println!("\n--- Prompt encoding tests ---");
    let test_prompts = [
        "Merhaba dünya",
        "Ankara ile Paris'in ortak özellikleri nelerdir?",
        "<|im_start|>user\nAnkara ile Paris'in ortak özellikleri nelerdir?\n<|im_end|>",
        "What are the common features of Ankara and Paris?",
        "<|im_start|>user\nWhat are the common features of Ankara and Paris?\n<|im_end|>",
        "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nAnkara ile Paris'in ortak özellikleri nelerdir?\n<|im_end|>\n<|im_start|>assistant\n<think>\n",
    ];

    for &prompt in &test_prompts {
        let enc = tok.encode(prompt, false).unwrap();
        let ids = enc.get_ids();
        let preview: String = prompt.chars().take(60).collect();
        println!("\n  Prompt: {}", preview);
        println!("  {} tokens: {:?}", ids.len(), ids);
        for (i, &id) in ids.iter().take(5).enumerate() {
            let dec = tok.decode(&[id], false).unwrap_or_default();
            println!("    [{}] id={} -> '{:?}'", i, id, dec);
        }
        if ids.len() > 5 {
            println!("    ... and {} more", ids.len() - 5);
        }
    }

    println!("\n--- Testing encode('<|im_start|>') with/without add_special_tokens ---");
    let no_special = tok.encode("<|im_start|>", false).unwrap();
    println!("  encode(\"<|im_start|>\", false): {:?}", no_special.get_ids());

    let with_special = tok.encode("<|im_start|>", true).unwrap();
    println!("  encode(\"<|im_start|>\", true):  {:?}", with_special.get_ids());
}
