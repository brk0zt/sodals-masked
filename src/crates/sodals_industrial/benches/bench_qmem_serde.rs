//! .qmem Serileştirme — Criterion Benchmark
//!
//! Metrikler:
//! - header_write: 32 byte header yazma süresi
//! - block_copy: 544 byte blok kopyalama (memcpy eşdeğeri)
//! - block_read: Struct'tan field okuma
//! - parse_10k: 10,000 kristel parse throughput

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use sodals_industrial::qmem_format::*;

fn make_block() -> FrequencySignatureBlock {
    let mut block = FrequencySignatureBlock {
        header: QmemHeader {
            magic: QmemHeader::MAGIC,
            version: [1, 0],
            dtype: 1,
            _pad0: 0,
            len: 128,
            sample_rate_hz: 256.0,
            timestamp_ns: 1_700_000_000_000_000_000u64,
            label: *b"BENCH___",
        },
        bands: [0.0f32; 128],
    };
    for i in 0..128 {
        block.bands[i] = -60.0 + (i as f32 * 0.5);
    }
    block
}

fn bench_header_write(c: &mut Criterion) {
    let h = QmemHeader {
        magic: QmemHeader::MAGIC,
        version: [1, 0],
        dtype: 1,
        _pad0: 0,
        len: 128,
        sample_rate_hz: 256.0,
        timestamp_ns: 0,
        label: *b"HEAD____",
    };
    let mut buf = [0u8; 32];

    c.bench_function("header_write_32bytes", |b| {
        b.iter(|| {
            h.write_to(&mut buf);
            black_box(&buf);
        });
    });
}

fn bench_block_copy(c: &mut Criterion) {
    let block = make_block();
    let mut buf = [0u8; 544];

    c.bench_function("block_memcpy_544bytes", |b| {
        b.iter(|| {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &block as *const _ as *const u8,
                    buf.as_mut_ptr(),
                    544,
                );
            }
            black_box(&buf);
        });
    });
}

fn bench_block_read_unaligned(c: &mut Criterion) {
    let block = make_block();
    let buf = unsafe {
        std::slice::from_raw_parts(
            &block as *const _ as *const u8,
            544,
        )
    };

    c.bench_function("block_read_unaligned_544bytes", |b| {
        b.iter(|| {
            let read: FrequencySignatureBlock = unsafe {
                std::ptr::read_unaligned(buf.as_ptr() as *const _)
            };
            black_box(read);
        });
    });
}

fn bench_parse_10k(c: &mut Criterion) {
    let block = make_block();
    let raw = unsafe {
        std::slice::from_raw_parts(&block as *const _ as *const u8, 544)
    };
    let mut buf = vec![0u8; 544 * 10_000];
    for i in 0..10_000 {
        buf[i * 544..(i + 1) * 544].copy_from_slice(raw);
    }

    let mut group = c.benchmark_group("qmem_parse");
    group.throughput(Throughput::Elements(10_000));

    group.bench_function("parse_10k_blocks", |b| {
        b.iter(|| {
            for i in 0..10_000 {
                let ptr = unsafe { buf.as_ptr().add(i * 544) };
                let read: FrequencySignatureBlock = unsafe {
                    std::ptr::read_unaligned(ptr as *const _)
                };
                black_box(read);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_header_write,
    bench_block_copy,
    bench_block_read_unaligned,
    bench_parse_10k
);
criterion_main!(benches);
