//! Ring Buffer — Criterion Benchmark
//!
//! Metrikler:
//! - push_latency: Tek push süresi (ns)
//! - pop_latency: Tek pop süresi (ns)
//! - pushpop_pair: Push+pop atomik süre (ns)
//! - spsc_throughput: 2 thread arası mesaj/sn

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sodals_industrial::RtRingBuffer;
use std::sync::Arc;
use std::thread;

fn bench_push_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_push_latency");
    for cap in [64, 256, 1024, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(cap), cap, |b, _cap| {
            let ring = RtRingBuffer::<u64, 1024>::new();
            b.iter(|| {
                black_box(ring.push(42u64));
            });
        });
    }
    group.finish();
}

fn bench_pop_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_pop_latency");
    for cap in [64, 256, 1024, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(cap), cap, |b, _cap| {
            let ring = RtRingBuffer::<u64, 1024>::new();
            // Pre-fill
            for i in 0..512 {
                let _ = ring.push(i);
            }
            b.iter(|| {
                black_box(ring.pop());
                let _ = ring.push(0u64);
            });
        });
    }
    group.finish();
}

fn bench_spsc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_spsc_throughput");
    group.throughput(Throughput::Elements(1_000_000));

    group.bench_function("1M_messages", |b| {
        b.iter_custom(|iters| {
            let ring = Arc::new(RtRingBuffer::<u64, 4096>::new());
            let ring_p = Arc::clone(&ring);
            let ring_c = Arc::clone(&ring);

            let start = std::time::Instant::now();

            thread::scope(|s| {
                s.spawn(move || {
                    for i in 0..iters {
                        while !ring_p.push(i) {
                            std::hint::spin_loop();
                        }
                    }
                });

                s.spawn(move || {
                    let mut count = 0u64;
                    while count < iters {
                        if ring_c.pop().is_some() {
                            count += 1;
                        }
                    }
                });
            });

            start.elapsed()
        });
    });

    group.finish();
}

fn bench_sensor_payload(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_sensor_payload");
    group.throughput(Throughput::Elements(100_000));

    group.bench_function("100K_sensor_messages", |b| {
        use sodals_industrial::SensorPayload;
        let ring = RtRingBuffer::<SensorPayload, 1024>::new();
        let payload = SensorPayload {
            timestamp_ns: 0,
            temperature: 25.0,
            pressure: 2.5,
            motor_rpm: 1500.0,
            status_mask: 0,
            fault_code: 0,
            _pad: [0; 2],
        };

        b.iter(|| {
            black_box(ring.push(payload));
            black_box(ring.pop());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_push_latency,
    bench_pop_latency,
    bench_spsc_throughput,
    bench_sensor_payload
);
criterion_main!(benches);
