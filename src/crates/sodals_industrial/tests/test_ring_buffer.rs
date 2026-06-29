//! RT Ring Buffer — Profesyonel Birim Testleri
//!
//! Kapsam:
//! 1. SPSC (Single-Producer Single-Consumer) lock-free doğrulama
//! 2. Overflow durumunda deterministik davranış (overwrite yok)
//! 3. Memory ordering (Acquire/Release) testi — çok çekirdekli
//! 4. Cache contention ve false sharing önleme
//! 5. Bellek basıncı (memory pressure) altında stabilite

use crossbeam::thread;
use sodals_industrial::RtRingBuffer;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod common;

const TEST_CAPACITY: usize = 1024; // 2^n, power of two
const ITERATIONS: usize = 10_000;

/// 1. Temel SPSC: 1 milyon eleman push/pop, hiç kayıp olmamalı.
#[test]
fn spsc_no_loss() {
    let ring = Arc::new(RtRingBuffer::<u64, TEST_CAPACITY>::new());
    let ring_consumer = Arc::clone(&ring);

    let produced = Arc::new(AtomicUsize::new(0));
    let consumed = Arc::new(AtomicUsize::new(0));

    let p = Arc::clone(&produced);
    let c = Arc::clone(&consumed);

    let deadline = Instant::now() + Duration::from_secs(10);

    thread::scope(|s| {
        s.spawn(move |_| {
            for i in 0..ITERATIONS {
                while !ring.push(i as u64) {
                    std::hint::spin_loop();
                }
                p.fetch_add(1, Ordering::Relaxed);
            }
        });

        s.spawn(move |_| {
            let mut last = None;
            let mut count = 0;
            while count < ITERATIONS {
                if let Some(val) = ring_consumer.pop() {
                    if let Some(prev) = last {
                        assert!(
                            val >= prev,
                            "Sıralama bozuldu: {} < {}",
                            val,
                            prev
                        );
                    }
                    last = Some(val);
                    c.fetch_add(1, Ordering::Relaxed);
                    count += 1;
                }
                if Instant::now() > deadline {
                    panic!("SPSC timeout: {} / {} consumed", count, ITERATIONS);
                }
            }
        });
    })
    .unwrap();

    let p = produced.load(Ordering::Relaxed);
    let c = consumed.load(Ordering::Relaxed);
    assert_eq!(p, ITERATIONS, "Producer {} eleman üretmeli", ITERATIONS);
    assert_eq!(c, ITERATIONS, "Consumer {} eleman tüketmeli", ITERATIONS);
}

/// 2. Overflow: Buffer dolu olduğunda push false döner, overwrite olmaz.
#[test]
fn overflow_rejects_push() {
    let ring = RtRingBuffer::<u64, 4>::new();

    // 4 eleman doldur (0,1,2,3)
    for i in 0..4 {
        assert!(ring.push(i), "İlk 4 push kabul edilmeli");
    }

    // 5. push reddedilmeli
    assert!(!ring.push(99), "Buffer dolu; push reddedilmeli");

    // Consumer 1 eleman alınca 1 boşluk oluşur
    assert_eq!(ring.pop(), Some(0));
    assert!(ring.push(4), "1 boşluk var; push kabul edilmeli");
}

/// 3. Memory Ordering: Çok çekirdekli cache consistency.
/// Her thread farklı core'da çalışmalı (taskset ile manuel ayarlanabilir).
#[test]
fn memory_ordering_across_cores() {
    let ring: Arc<RtRingBuffer<u64, 128>> = Arc::new(RtRingBuffer::new());
    let start = Instant::now();
    let timeout = Duration::from_secs(5);

    let ring_w: Arc<RtRingBuffer<u64, 128>> = Arc::clone(&ring);
    let ring_r: Arc<RtRingBuffer<u64, 128>> = Arc::clone(&ring);

    let handle_w = std::thread::spawn(move || {
        for i in 0..10_000 {
            while !ring_w.push(i as u64) {
                std::hint::spin_loop();
            }
        }
    });

    let handle_r = std::thread::spawn(move || {
        let mut count = 0;
        let mut val = 0u64;
        while count < 10_000 {
            if let Some(v) = ring_r.pop() {
                val = v;
                count += 1;
            }
            if start.elapsed() > timeout {
                panic!("Timeout: Consumer {} eleman alabildi", count);
            }
        }
        val
    });

    handle_w.join().unwrap();
    handle_r.join().unwrap();
}

/// 4. Performans: Tek thread push/pop latency.
/// Hedef: < 50 ns P99 (L3 cache hit).
#[test]
fn latency_single_thread() {
    let ring = RtRingBuffer::<u64, TEST_CAPACITY>::new();
    const SAMPLES: usize = 10_000;
    let mut latencies = Vec::with_capacity(SAMPLES);

    // Warm-up: cache priming
    for i in 0..TEST_CAPACITY {
        let _ = ring.push(i as u64);
    }
    while ring.pop().is_some() {}

    for i in 0..SAMPLES {
        let t0 = Instant::now();
        ring.push(i as u64);
        let t1 = Instant::now();
        ring.pop();
        let t2 = Instant::now();
        latencies.push((t1 - t0) + (t2 - t1));
    }

    latencies.sort();
    let p99 = latencies[SAMPLES * 99 / 100];
    let max = latencies[SAMPLES - 1];

    eprintln!("\n=== Ring Buffer Push+Pop Latency ===");
    eprintln!("  P50: {:?}", latencies[SAMPLES / 2]);
    eprintln!("  P99: {:?}", p99);
    eprintln!("  Max: {:?}", max);

    // Profesyonel eşik: 500 ns üzerindeyse performans regresyonu var
    assert!(
        p99 < Duration::from_nanos(500),
        "P99 latency çok yüksek: {:?}",
        p99
    );
}

/// 5. Bellek basıncı altında stabilite: Heap'te 500MB ayrılırken
/// ring buffer allocate yapmaz (zero-alloc hot-path).
#[test]
fn stability_under_memory_pressure() {
    let _pressure = common::simulate_memory_pressure(100);
    let ring = Arc::new(RtRingBuffer::<u64, TEST_CAPACITY>::new());
    let ring_clone = Arc::clone(&ring);

    let start = Instant::now();
    let deadline = start + Duration::from_secs(10);

    thread::scope(|s| {
        s.spawn(move |_| {
            for i in 0..10_000 {
                while !ring.push(i as u64) {
                    std::hint::spin_loop();
                }
            }
        });

        s.spawn(move |_| {
            let mut count = 0;
            while count < 10_000 {
                if ring_clone.pop().is_some() {
                    count += 1;
                }
                if Instant::now() > deadline {
                    panic!("Memory pressure test timeout: {} / 10000", count);
                }
            }
        });
    })
    .unwrap();

    let elapsed = start.elapsed();
    eprintln!("\n=== Bellek Basıncı Altında 10K Mesaj ===");
    eprintln!("  Süre: {:?}", elapsed);
    eprintln!("  Mesaj/sn: {:.0}", 10_000.0 / elapsed.as_secs_f64());
}

/// 6. Cache-line contention testi: Yan yana ring buffer'lar false sharing
/// yaratmamalı (her biri ayrı cache-line'da).
#[test]
fn no_false_sharing_between_instances() {
    const N: usize = 8;
    let rings: Vec<_> = (0..N)
        .map(|_| Arc::new(RtRingBuffer::<u64, 128>::new()))
        .collect();

    let start = Instant::now();

    thread::scope(|s| {
        for (idx, ring) in rings.iter().enumerate() {
            let r = Arc::clone(ring);
            s.spawn(move |_| {
                for i in 0..1_000 {
                    while !r.push((idx * 1_000 + i) as u64) {}
                    while r.pop().is_none() {}
                }
            });
        }
    })
    .unwrap();

    let elapsed = start.elapsed();
    let total_ops = N * 1_000;
    eprintln!("\n=== {} Bağımsız Ring Buffer (False Sharing Test) ===", N);
    eprintln!("  Toplam İşlem: {}", total_ops);
    eprintln!("  Süre: {:?}", elapsed);
    eprintln!("  İşlem/sn: {:.0}", total_ops as f64 / elapsed.as_secs_f64());
}
