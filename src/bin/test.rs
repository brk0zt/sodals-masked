fn main() {
    let mut varmap = candle_nn::VarMap::new();
    let data = varmap.data().lock().unwrap();
}
