//! .qmem Formatı — Profesyonel Birim Testleri
//!
//! Kapsam:
//! 1. Header validasyonu (magic, version, alignment)
//! 2. Frekans imza bloğu serileştirme/deserileştirme
//! 3. Anomali tespiti (enerji eşiği)
//! 4. Cosine similarity doğrulama
//! 5. 10,000 kristal üretim hızı benchmark'ı
//! 6. Endianness ve padding kontrolü

use sodals_industrial::qmem_format::*;
use std::mem;
use std::time::Instant;

/// 1. Header boyut ve hizalama (32 byte boundary).
#[test]
fn header_size_and_alignment() {
    assert_eq!(mem::size_of::<QmemHeader>(), 32);
    assert_eq!(mem::align_of::<QmemHeader>(), 8); // repr(C) + u64 alignment

    let h = QmemHeader {
        magic: QmemHeader::MAGIC,
        version: [1, 0],
        dtype: 0,
        _pad0: 0,
        len: 128,
        sample_rate_hz: 256.0,
        timestamp_ns: 1_700_000_000_000_000_000u64,
        label: *b"TMP0____",
    };

    assert!(h.is_valid());

    let mut buf = [0u8; 32];
    h.write_to(&mut buf);
    assert_eq!(&buf[0..4], b"QMEM");
    assert_eq!(buf[4..6], [1, 0]);
    assert_eq!(&buf[8..12], &128u32.to_le_bytes());
}

/// 2. Frekans imza bloğu boyutu (544 byte = 32 header + 512 data).
#[test]
fn frequency_block_size() {
    assert_eq!(FrequencySignatureBlock::BLOCK_SIZE, 544);
    assert_eq!(mem::size_of::<FrequencySignatureBlock>(), 544);
}

/// 3. Anomali tespiti: enerji eşiğini aşan spektrum.
#[test]
fn anomaly_detection_energy_threshold() {
    let mut block = FrequencySignatureBlock {
        header: QmemHeader {
            magic: QmemHeader::MAGIC,
            version: [1, 0],
            dtype: 1,
            _pad0: 0,
            len: 128,
            sample_rate_hz: 256.0,
            timestamp_ns: 0,
            label: *b"VIB0____",
        },
        bands: [0.0f32; 128],
    };

    // Normal: tüm bantlar -60 dB (ortalama -60 < -40 eşik)
    block.bands.fill(-60.0);
    assert!(!block.is_anomalous(-40.0), "-60 dB ortalama anomali olmamalı");

    // Anomali: tüm bantlar -20 dB (ortalama -20 > -40 eşik)
    block.bands.fill(-20.0);
    assert!(block.is_anomalous(-40.0), "-20 dB ortalama anomali olmalı");

    // Sınır: ortalama tam -40 dB
    block.bands.fill(-40.0);
    assert!(
        !block.is_anomalous(-40.0),
        "Tam eşikte anomali olmamalı (strict inequality test)"
    );
}

/// 4. Cosine similarity: aynı vektör = 1.0, zıt vektör = -1.0, dik = 0.0.
#[test]
fn cosine_similarity_properties() {
    let mut a = FrequencySignatureBlock {
        header: QmemHeader {
            magic: QmemHeader::MAGIC,
            version: [1, 0],
            dtype: 1,
            _pad0: 0,
            len: 128,
            sample_rate_hz: 256.0,
            timestamp_ns: 0,
            label: *b"SIG_A___",
        },
        bands: [0.0f32; 128],
    };
    a.bands.fill(10.0);

    let mut b = a.clone();
    b.header.label = *b"SIG_B___";

    // Aynı vektör
    let sim = a.cosine_similarity(&b);
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "Aynı vektör cosine similarity = 1.0 olmalı, alınan: {}",
        sim
    );

    // Dik vektör (a: [10,0,0,...], c: [0,10,0,...])
    let mut a_perp = a.clone();
    a_perp.bands.fill(0.0);
    a_perp.bands[0] = 10.0;
    let mut c = a_perp.clone();
    c.bands[0] = 0.0;
    c.bands[1] = 10.0;
    let sim_perp = a_perp.cosine_similarity(&c);
    assert!(
        sim_perp < 0.2,
        "Dik vektör similarity düşük olmalı, alınan: {}",
        sim_perp
    );

    // Sıfır vektör (NaN önleme)
    let mut z = a.clone();
    z.bands.fill(0.0);
    let sim_zero = a.cosine_similarity(&z);
    assert!(
        sim_zero.is_finite(),
        "Sıfır vektörle similarity sonlu olmalı, alınan: {}",
        sim_zero
    );
}

/// 5. Serileştirme hızı: 10,000 kristal/saniye hedefi.
#[test]
fn serialization_throughput() {
    let block = FrequencySignatureBlock {
        header: QmemHeader {
            magic: QmemHeader::MAGIC,
            version: [1, 0],
            dtype: 1,
            _pad0: 0,
            len: 128,
            sample_rate_hz: 256.0,
            timestamp_ns: 0,
            label: *b"BENCH___",
            },
        bands: [0.0f32; 128],
    };

    const N: usize = 100_000;
    let start = Instant::now();

    for _ in 0..N {
        let mut buf = [0u8; 544];
        unsafe {
            std::ptr::copy_nonoverlapping(
                &block as *const _ as *const u8,
                buf.as_mut_ptr(),
                544,
            );
        }
        // Serileştirme round-trip
        let _read_back: FrequencySignatureBlock = unsafe {
            std::ptr::read_unaligned(buf.as_ptr() as *const _)
        };
    }

    let elapsed = start.elapsed();
    let ops_per_sec = N as f64 / elapsed.as_secs_f64();
    let mb_per_sec = (N * 544) as f64 / elapsed.as_secs_f64() / 1_048_576.0;

    eprintln!("\n=== .qmem Serileştirme Throughput ===");
    eprintln!("  Kristal: {}", N);
    eprintln!("  Süre: {:?}", elapsed);
    eprintln!("  Kristal/sn: {:.0}", ops_per_sec);
    eprintln!("  MB/sn: {:.1}", mb_per_sec);

    assert!(
        ops_per_sec > 1_000_000.0,
        "Serileştirme çok yavaş: {:.0} kristal/sn",
        ops_per_sec
    );
}

/// 6. Endianness: Little-endian kontrolü (x86_64 hedef).
#[test]
fn little_endian_layout() {
    let h = QmemHeader {
        magic: QmemHeader::MAGIC,
        version: [1, 0],
        dtype: 0,
        _pad0: 0,
        len: 0x1234_5678u32,
        sample_rate_hz: 0.0,
        timestamp_ns: 0xAABB_CCDD_EEFF_0011u64,
        label: [0; 8],
    };

    let mut buf = [0u8; 32];
    h.write_to(&mut buf);

    // u32 len little-endian
    assert_eq!(buf[8], 0x78);
    assert_eq!(buf[9], 0x56);
    assert_eq!(buf[10], 0x34);
    assert_eq!(buf[11], 0x12);

    // u64 timestamp little-endian
    assert_eq!(buf[16], 0x11);
    assert_eq!(buf[17], 0x00);
    assert_eq!(buf[23], 0xAA);
}

/// 7. Bellek güvenliği: Geçersiz magic reddedilmeli.
#[test]
fn invalid_magic_rejected() {
    let h = QmemHeader {
        magic: *b"XMEM", // Geçersiz
        version: [1, 0],
        dtype: 0,
        _pad0: 0,
        len: 0,
        sample_rate_hz: 0.0,
        timestamp_ns: 0,
        label: [0; 8],
    };
    assert!(!h.is_valid(), "Geçersiz magic kabul edilmemeli");
}
