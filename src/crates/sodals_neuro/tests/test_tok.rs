#[test]
fn try_parse() {
    let path = r#"c:\Users\Burak\Desktop\sodals_core\models\tokenizer.json"#;
    let s = std::fs::read_to_string(path).unwrap();
    let err = match s.parse::<tokenizers::Tokenizer>() {
        Ok(_) => "Success".to_string(),
        Err(e) => e.to_string()
    };
    println!("PARSE RESULT: {}", err);
}













