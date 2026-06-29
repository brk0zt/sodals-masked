use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sodals_core::semantic_network::ResonanceNetwork;
use num_complex::Complex64;

fn benchmark_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("ag_simulation");
    
    // Scenario: 10,000 Neurons, 50,000 Connections
    let mut network = ResonanceNetwork::new(10_000);
    
    // 1. Add Neurons
    for _ in 0..10_000 {
        network.add_concept(0.1);
    }

    // 2. Add Connections
    // We use a mix of Syntax (1 edge) and Semantic (2 edges) to reach ~50k edges.
    // 40,000 Syntax Edges
    for i in 0..40_000 {
        let src = i % 10_000;
        let tgt = (i + 1) % 10_000;
        network.connect_syntax(src, tgt, 1.0);
    }
    
    // 5,000 Semantic Connections (x2 edges each = 10,000 edges)
    for i in 0..5_000 {
        let src = i % 10_000;
        let tgt = (i + 50) % 10_000;
        network.connect_semantic(src, tgt, 0.5);
    }

    let inputs = vec![(0, Complex64::new(1.0, 0.0))];

    group.bench_function("step_10k_neurons", |b| {
        b.iter(|| {
            network.step(black_box(&inputs))
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_step);
criterion_main!(benches);