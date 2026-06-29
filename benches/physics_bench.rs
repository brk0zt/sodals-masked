// benches/physics_bench.rs - CRITERION PERFORMANCE BENCHMARKS

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use num_complex::Complex64;
use sodals_neuro::fast_math::SemanticWave;
use sodals_neuro::semantic_network::ResonanceNetwork;

/// Benchmark: SemanticWave SIMD operations vs naive loops
fn benchmark_quantum_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_state");
    
    for size in [10, 100, 1000, 10000].iter() {
        let a = SemanticWave::from_vec(vec![Complex64::new(1.0, 0.5); *size]);
        let b = SemanticWave::from_vec(vec![Complex64::new(0.5, 1.0); *size]);

        group.bench_with_input(
            BenchmarkId::new("simd_addition", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(a.clone() + b.clone());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_inner_product", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(a.inner_product(&b));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_norm_sqr", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(a.norm_sqr());
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Legacy O(E) vs Matrix-based O(N²) force calculation
fn benchmark_force_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("force_computation");
    
    for n in [10, 50, 100, 500].iter() {
        let mut net = ResonanceNetwork::new(*n);
        
        // Create fully connected network (worst case)
        for i in 0..*n {
            net.add_concept(0.05);
        }
        
        for i in 0..*n {
            for j in (i+1)..*n {
                net.connect_semantic(i, j, 0.3);
            }
        }

        // Initialize with random state
        for i in 0..*n {
            net.positions.amplitudes[i] = Complex64::new(
                (i as f64).sin(),
                (i as f64).cos(),
            );
        }

        group.bench_with_input(
            BenchmarkId::new("matrix_forces", n),
            n,
            |bench, _| {
                bench.iter(|| {
                    black_box(net.compute_forces_sparse(&[]));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Full symplectic integration step
fn benchmark_integration_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("symplectic_integration");
    
    for n in [10, 100, 500, 1000].iter() {
        let mut net = ResonanceNetwork::new(*n);
        
        // Create realistic sparse network (10% connectivity)
        for i in 0..*n {
            net.add_concept(0.02);
        }
        
        for i in 0..*n {
            for j in (i+1)..*n {
                if (i * 7 + j * 13) % 10 == 0 {
                    net.connect_semantic(i, j, 0.4);
                }
            }
        }

        let external = vec![(0, Complex64::new(1.0, 0.0))];

        group.bench_with_input(
            BenchmarkId::new("single_step", n),
            n,
            |bench, _| {
                bench.iter(|| {
                    black_box(net.step(&external));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Multi-step simulation (1000 steps)
fn benchmark_long_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("long_simulation");
    group.sample_size(10); // Reduce sample size for long benchmarks
    
    for n in [10, 50, 100].iter() {
        let mut net = ResonanceNetwork::new(*n);
        
        for i in 0..*n {
            net.add_concept(0.05);
        }
        
        // Chain topology (realistic for semantic networks)
        for i in 0..(*n - 1) {
            net.connect_semantic(i, i + 1, 0.5);
        }

        let external = vec![(0, Complex64::new(1.0, 0.0))];

        group.bench_with_input(
            BenchmarkId::new("1000_steps", n),
            n,
            |bench, _| {
                bench.iter(|| {
                    for _ in 0..1000 {
                        black_box(net.step(&external));
                    }
                    net.reset();
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Weighted sum for network output
fn benchmark_weighted_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_sum");
    
    for n in [10, 100, 1000].iter() {
        let states: Vec<SemanticWave> = (0..*n)
            .map(|i| {
                SemanticWave::from_vec(vec![
                    Complex64::new((i as f64).sin(), (i as f64).cos());
                    100
                ])
            })
            .collect();
        
        let weights: Vec<f64> = (0..*n).map(|i| (i as f64) / (*n as f64)).collect();

        group.bench_with_input(
            BenchmarkId::new("vectorized", n),
            n,
            |bench, _| {
                bench.iter(|| {
                    black_box(SemanticWave::weighted_sum(&states, &weights));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Energy conservation calculation overhead
fn benchmark_energy_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy_calculation");
    
    for n in [10, 100, 500].iter() {
        let mut net = ResonanceNetwork::new(*n);
        
        for i in 0..*n {
            net.add_concept(0.03);
        }
        
        for i in 0..*n {
            for j in (i+1)..*n {
                if (i + j) % 5 == 0 {
                    net.connect_semantic(i, j, 0.4);
                }
            }
        }

        // Initialize state
        for i in 0..*n {
            net.positions.amplitudes[i] = Complex64::new(i as f64, 0.0);
            net.momenta.amplitudes[i] = Complex64::new(0.0, i as f64);
        }

        group.bench_with_input(
            BenchmarkId::new("hamiltonian", n),
            n,
            |bench, _| {
                bench.iter(|| {
                    black_box(net.calculate_energy());
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_quantum_operations,
    benchmark_force_computation,
    benchmark_integration_step,
    benchmark_long_simulation,
    benchmark_weighted_sum,
    benchmark_energy_calculation,
);

criterion_main!(benches);