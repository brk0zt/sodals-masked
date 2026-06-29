//! Sentinel Guard — Criterion Benchmark
//!
//! Metrikler:
//! - evaluate_latency: Tek karar süresi (ns)
//! - cosine_similarity: İki 128-bantlı vektör karşılaştırması (ns)
//! - anomaly_detection: FFT enerji hesabı (ns)
//! - batch_1M: 1 milyon kristel toplu işleme throughput

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use sodals_industrial::{
    qmem_format::*,
    sentinel_guard::*,
};

fn make_guard() -> SentinelGuard {
    let mut sig = FrequencySignatureBlock {
        header: QmemHeader {
            magic: QmemHeader::MAGIC,
            version: [1, 0],
            dtype: 1,
            _pad0: 0,
            len: 128,
            sample_rate_hz: 256.0,
            timestamp_ns: 0,
            label: *b"REF_____",
        },
        bands: [-50.0f32; 128],
    };
    sig.bands[10] = -30.0;
    sig.bands[20] = -35.0;

    SentinelGuard::new(
        SentinelReference {
            signature: sig,
            threshold_db: -40.0,
        },
        3,
    )
}

fn bench_evaluate(c: &mut Criterion) {
    let guard = make_guard();
    let mut candidate_block = FrequencySignatureBlock {
        header: QmemHeader {
            magic: QmemHeader::MAGIC,
            version: [1, 0],
            dtype: 1,
            _pad0: 0,
            len: 128,
            sample_rate_hz: 256.0,
            timestamp_ns: 0,
            label: *b"CAND____",
        },
        bands: [-50.0f32; 128],
    };

    let mut group = c.benchmark_group("sentinel_evaluate");
    group.bench_function("normal_allow", |b| {
        b.iter(|| {
            black_box(guard.evaluate(&candidate_block));
        });
    });

    // Anomali durumu
    candidate_block.bands.fill(-10.0);
    group.bench_function("anomaly_reject", |b| {
        b.iter(|| {
            black_box(guard.evaluate(&candidate_block));
        });
    });

    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut a = FrequencySignatureBlock {
        header: QmemHeader {
            magic: QmemHeader::MAGIC,
            version: [1, 0],
            dtype: 1,
            _pad0: 0,
            len: 128,
            sample_rate_hz: 256.0,
            timestamp_ns: 0,
            label: *b"A_______",
        },
        bands: [0.0f32; 128],
    };
    a.bands.fill(10.0);

    let mut other = a.clone();
    other.header.label = *b"B_______";

    c.bench_function("cosine_similarity_128bands", |bencher| {
        bencher.iter(|| {
            black_box(a.cosine_similarity(&other));
        });
    });
}

fn bench_batch_1m(c: &mut Criterion) {
    let guard = make_guard();
    let mut candidate = FrequencySignatureBlock {
        header: QmemHeader {
            magic: QmemHeader::MAGIC,
            version: [1, 0],
            dtype: 1,
            _pad0: 0,
            len: 128,
            sample_rate_hz: 256.0,
            timestamp_ns: 0,
            label: *b"BATCH___",
        },
        bands: [-50.0f32; 128],
    };

    let mut group = c.benchmark_group("sentinel_batch");
    group.throughput(Throughput::Elements(1_000_000));

    group.bench_function("1M_evaluations", |b| {
        b.iter(|| {
            for i in 0..1_000_000usize {
                candidate.header.timestamp_ns = i as u64;
                black_box(guard.evaluate(&candidate));
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_evaluate, bench_cosine_similarity, bench_batch_1m);
criterion_main!(benches);
