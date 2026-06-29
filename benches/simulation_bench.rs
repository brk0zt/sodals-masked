// benches/simulation_bench.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sodals_core::semantic_network::ResonanceNetwork; // Proje adın neyse onu yaz (Cargo.toml'daki name)
use num_complex::Complex64;

fn benchmark_step(c: &mut Criterion) {
    let mut network = ResonanceNetwork::new(1000);
    
    // 1000 Nöronluk Ağ Kur
    for _ in 0..1000 {
        network.add_concept(0.1);
    }
    
    // Rastgele 5000 bağlantı yap
    for i in 0..900 {
        network.connect_semantic(i, i+1, 0.5);
        network.connect_syntax(i, (i+50) % 1000, 1.0);
    }

    let inputs = vec![];

    // Performans Testi
    c.bench_function("network_step_1000_neurons", |b| {
        b.iter(|| {
            // black_box, derleyicinin kodu optimize edip silmesini engeller
            network.step(black_box(&inputs))
        })
    });
}

criterion_group!(benches, benchmark_step);
criterion_main!(benches);