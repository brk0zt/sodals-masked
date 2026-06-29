use tokenizers::Tokenizer;

fn main() {
    let path = r#"c:\Users\Burak\Desktop\sodals_core\models\tokenizer.json"#;
    let s = std::fs::read_to_string(path).unwrap();
    let err = match Tokenizer::from_str(&s) {
        Ok(_) => "Success".to_string(),
        Err(e) => e.to_string()
    };
    println!("ORIGINAL: {}", err);
}















