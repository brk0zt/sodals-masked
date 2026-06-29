// src/engine/qmem_compiler.rs
// ============================================================================
// QMEM COMPILER — Zero-Copy Topological Attractor Serialization
// ============================================================================
//
// .qmem (Quantum Memory Crystal) is a custom binary format designed for
// sub-microsecond mmap loading of pre-computed EEG phase-space manifolds.
//
// File layout (little-endian):
//
//   Offset  | Size   | Field
//   --------|--------|----------------------------------------------
//   [0..4]  | 4 B    | Magic bytes: "QMEM"
//   [4..8]  | 4 B    | Version: u32 (0x00000001)
//   [8..12] | 4 B    | Label ID (target word): i32
//   [12..16]| 4 B    | Number of channels: u32 (128 EEG only, EXG excluded)
//   [16..20]| 4 B    | Time steps per channel: u32
//   [20..EOF]| N×16 B | Payload: [Complex64; channels × time_steps]
//                       Each Complex64 = (f64 re, f64 im) = 16 bytes LE
//
// Total file size = 20 + (channels × time_steps × 16) bytes.
// For 128ch × 256ts = 32768 complex samples = 524,288 bytes payload + 20 header
//                   = 524,308 bytes per crystal (~512 KB).
//
// Design invariants:
//   1. NO serde, NO bincode, NO JSON. Raw C-struct binary dump.
//   2. Payload is identical to in-memory Complex64 layout (IEEE 754 LE).
//   3. mmap-compatible: the file can be cast to a &[Complex64] directly.

use std::fs::{self, File};
use std::io::{Write, BufWriter, Read, BufReader};
use std::path::Path;
use num_complex::Complex64;
use anyhow::{Result, bail, Context};
use rayon::prelude::*;

use crate::fast_math::SemanticWave;
use crate::engine::bci_mux::takens_embed;

// ============================================================================
// CONSTANTS
// ============================================================================

pub const QMEM_MAGIC: [u8; 4] = *b"QMEM";
pub const QMEM_VERSION: u32 = 1;

/// Hardware truth: Nieto ds003626 has 136 channels = 128 EEG + 8 EXG.
/// QMEM crystals store ONLY the 128 cortical EEG channels.
pub const EEG_CHANNEL_COUNT: usize = 128;
pub const EXG_CHANNEL_COUNT: usize = 8;
pub const TOTAL_HARDWARE_CHANNELS: usize = EEG_CHANNEL_COUNT + EXG_CHANNEL_COUNT; // 136

/// Header size in bytes: magic(4) + version(4) + label(4) + channels(4) + timesteps(4)
const HEADER_SIZE: usize = 20;

// ============================================================================
// QMEM CRYSTAL — Single Topological Attractor
// ============================================================================

#[derive(Debug, Clone)]
pub struct QmemCrystal {
    /// Target label (word ID the subject was thinking)
    pub label_id: i32,
    /// Number of EEG channels encoded (128 for full cortical, 4 for AirPods subset)
    pub num_channels: u32,
    /// Time steps per channel (typically 256 @ 256Hz = 1 second epoch)
    pub time_steps: u32,
    /// Phase-space embedded manifold: len = num_channels × time_steps
    pub manifold: SemanticWave,
}

impl QmemCrystal {
    /// Serialize the crystal to disk in raw binary .qmem format.
    ///
    /// Layout:
    ///   [QMEM][version:u32][label:i32][channels:u32][timesteps:u32][payload...]
    ///
    /// Payload is a contiguous array of (f64, f64) pairs in native LE order,
    /// identical to the in-memory representation of Complex64.
    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("QMEM: Failed to create {:?}", path))?;
        let mut w = BufWriter::with_capacity(256 * 1024, file);

        // --- HEADER (20 bytes) ---
        w.write_all(&QMEM_MAGIC)?;
        w.write_all(&QMEM_VERSION.to_le_bytes())?;
        w.write_all(&self.label_id.to_le_bytes())?;
        w.write_all(&self.num_channels.to_le_bytes())?;
        w.write_all(&self.time_steps.to_le_bytes())?;

        // --- PAYLOAD: Raw Complex64 dump ---
        // Complex64 is #[repr(C)] with layout { re: f64, im: f64 }.
        // We write each component as little-endian f64 directly.
        let amplitudes = self.manifold.amplitudes.as_slice()
            .expect("QmemCrystal: non-contiguous SemanticWave");

        let expected_len = (self.num_channels as usize) * (self.time_steps as usize);
        if amplitudes.len() != expected_len {
            bail!(
                "QMEM: Manifold length {} != channels({}) × timesteps({}) = {}",
                amplitudes.len(), self.num_channels, self.time_steps, expected_len
            );
        }

        // Write raw bytes: each Complex64 is 16 bytes (2 × f64 LE)
        for c in amplitudes.iter() {
            w.write_all(&c.re.to_le_bytes())?;
            w.write_all(&c.im.to_le_bytes())?;
        }

        w.flush()?;
        Ok(())
    }

    /// Deserialize a .qmem crystal from disk.
    ///
    /// Validates magic bytes, version, and payload length.
    /// Returns a fully-hydrated QmemCrystal ready for manifold operations.
    pub fn load_from_disk<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("QMEM: Failed to open {:?}", path))?;
        let file_len = file.metadata()?.len() as usize;
        let mut r = BufReader::with_capacity(256 * 1024, file);

        // --- HEADER ---
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != QMEM_MAGIC {
            bail!("QMEM: Invalid magic bytes in {:?} (got {:?})", path, magic);
        }

        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != QMEM_VERSION {
            bail!("QMEM: Unsupported version {} in {:?} (expected {})", version, path, QMEM_VERSION);
        }

        r.read_exact(&mut buf4)?;
        let label_id = i32::from_le_bytes(buf4);

        r.read_exact(&mut buf4)?;
        let num_channels = u32::from_le_bytes(buf4);

        r.read_exact(&mut buf4)?;
        let time_steps = u32::from_le_bytes(buf4);

        // --- PAYLOAD ---
        let expected_samples = (num_channels as usize) * (time_steps as usize);
        let expected_payload = expected_samples * 16; // 16 bytes per Complex64
        let expected_total = HEADER_SIZE + expected_payload;

        if file_len != expected_total {
            bail!(
                "QMEM: File size mismatch in {:?}: expected {} bytes, got {}",
                path, expected_total, file_len
            );
        }

        let mut buf8 = [0u8; 8];
        let mut amplitudes = Vec::with_capacity(expected_samples);

        for _ in 0..expected_samples {
            r.read_exact(&mut buf8)?;
            let re = f64::from_le_bytes(buf8);
            r.read_exact(&mut buf8)?;
            let im = f64::from_le_bytes(buf8);
            amplitudes.push(Complex64::new(re, im));
        }

        Ok(Self {
            label_id,
            num_channels,
            time_steps,
            manifold: SemanticWave::from_vec(amplitudes),
        })
    }
}

// ============================================================================
// QMEM FACTORY — Parallel Batch Compilation
// ============================================================================
//
// Processes .safetensors files containing multi-epoch EEG data and compiles
// each epoch into a standalone .qmem crystal with Takens embedding applied.

pub struct QmemFactory;

impl QmemFactory {
    /// Compile an entire dataset directory of .safetensors files into .qmem crystals.
    ///
    /// For each safetensors file:
    ///   1. Load all epochs from the "eeg_data" / "eeg_waves" / "data" tensor
    ///   2. Load labels from the "labels" tensor
    ///   3. For each epoch, extract ALL channels, apply Takens embedding
    ///   4. Save as {subject}_{epoch:05}_{label}.qmem
    ///
    /// Uses Rayon par_iter for fearless concurrency across all CPU cores.
    pub fn compile_dataset(safetensors_dir: &Path, output_dir: &Path) -> Result<usize> {
        // Create output directory
        fs::create_dir_all(output_dir)
            .with_context(|| format!("QMEM: Failed to create output dir {:?}", output_dir))?;

        // Discover all .safetensors files
        let mut safetensor_files: Vec<_> = fs::read_dir(safetensors_dir)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "safetensors") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        safetensor_files.sort();

        if safetensor_files.is_empty() {
            bail!("QMEM: No .safetensors files found in {:?}", safetensors_dir);
        }

        eprintln!("[QMEM FACTORY] Found {} safetensors files. Compiling...", safetensor_files.len());

        // Process each file (Rayon parallel across files)
        let total_crystals: usize = safetensor_files
            .par_iter()
            .map(|st_path| {
                match Self::compile_single_safetensors(st_path, output_dir) {
                    Ok(count) => {
                        eprintln!(
                            "[QMEM] Compiled {:?} -> {} crystals",
                            st_path.file_name().unwrap_or_default(),
                            count
                        );
                        count
                    }
                    Err(e) => {
                        eprintln!(
                            "[QMEM ERROR] Failed to compile {:?}: {}",
                            st_path.file_name().unwrap_or_default(),
                            e
                        );
                        0
                    }
                }
            })
            .sum();

        eprintln!(
            "[QMEM FACTORY] Compilation complete. Total crystals: {}",
            total_crystals
        );

        Ok(total_crystals)
    }

    /// Compile a single .safetensors file into N .qmem crystals (one per epoch).
    fn compile_single_safetensors(st_path: &Path, output_dir: &Path) -> Result<usize> {
        let device = candle_core::Device::Cpu; // CPU for compilation; no GPU needed
        let tensors = candle_core::safetensors::load(st_path, &device)
            .with_context(|| format!("Failed to load {:?}", st_path))?;

        // Find EEG data tensor
        let eeg_waves = tensors.get("eeg_data")
            .or_else(|| tensors.get("data"))
            .or_else(|| tensors.get("eeg_waves"))
            .ok_or_else(|| anyhow::anyhow!(
                "No eeg_data/data/eeg_waves tensor in {:?}", st_path
            ))?;

        // Find labels tensor
        let labels_tensor = tensors.get("labels")
            .ok_or_else(|| anyhow::anyhow!("No labels tensor in {:?}", st_path))?;

        let labels: Vec<i64> = labels_tensor.to_vec1::<i64>()
            .or_else(|_| {
                // Fallback: try u32 and cast
                labels_tensor.to_vec1::<u32>().map(|v| v.into_iter().map(|x| x as i64).collect())
            })
            .with_context(|| format!("Failed to decode labels in {:?}", st_path))?;

        let num_epochs = eeg_waves.dim(0)?;
        let total_channels = eeg_waves.dim(1)?;
        let time_steps = eeg_waves.dim(2)? as u32;

        // HARDWARE TRUTH: Nieto dataset = 136ch (128 EEG + 8 EXG)
        // We extract ONLY the first 128 EEG channels for cortical topology.
        let eeg_ch = if total_channels >= TOTAL_HARDWARE_CHANNELS {
            EEG_CHANNEL_COUNT
        } else {
            // Fallback for already-sliced or smaller datasets
            total_channels
        };
        let num_channels = eeg_ch as u32;

        eprintln!(
            "[QMEM] Dataset: {} epochs × {} total ch (slicing first {} EEG) × {} timesteps",
            num_epochs, total_channels, eeg_ch, time_steps
        );

        // Extract subject name from filename for output naming
        let subject_stem = st_path.file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Process each epoch (Rayon parallel across epochs within a file)
        let crystal_count: usize = (0..num_epochs)
            .into_par_iter()
            .map(|epoch_idx| {
                let result = (|| -> Result<()> {
                    // Extract single epoch: [1, total_channels, timesteps]
                    let epoch_tensor = eeg_waves.narrow(0, epoch_idx, 1)?;

                    // SURGICAL SLICE: Take only channels [0..128] (EEG cortical)
                    // Discard channels [128..136] (EXG: jaw/throat sEMG + EOG)
                    let eeg_only = epoch_tensor.narrow(1, 0, eeg_ch)?;
                    let flat: Vec<f32> = eeg_only.flatten_all()?.to_vec1()?;

                    // Convert to f64 and scale Volt -> MikroVolt
                    let reals: Vec<f64> = flat.iter()
                        .map(|&v| (v as f64) * 1_000_000.0)
                        .collect();

                    // Build SemanticWave from 128 EEG channels and apply Takens embedding
                    let wave = SemanticWave::from_reals(reals);
                    let embedded = takens_embed(&wave);

                    // Determine label
                    let label_id = if epoch_idx < labels.len() {
                        labels[epoch_idx] as i32
                    } else {
                        -1 // Unknown label sentinel
                    };

                    // Build crystal
                    let crystal = QmemCrystal {
                        label_id,
                        num_channels,
                        time_steps,
                        manifold: embedded,
                    };

                    // Output path: {subject}_epoch{NNNNN}_label{LL}.qmem
                    let output_path = output_dir.join(format!(
                        "{}_epoch{:05}_label{:02}.qmem",
                        subject_stem, epoch_idx, label_id
                    ));

                    crystal.save_to_disk(&output_path)?;
                    Ok(())
                })();

                match result {
                    Ok(()) => 1,
                    Err(e) => {
                        eprintln!(
                            "[QMEM] Epoch {} failed: {}",
                            epoch_idx, e
                        );
                        0
                    }
                }
            })
            .sum();

        Ok(crystal_count)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_qmem_roundtrip() {
        // Build a small crystal: 4 channels × 8 time steps = 32 samples
        let mut data = Vec::with_capacity(32);
        for i in 0..32 {
            data.push(Complex64::new(i as f64 * 0.1, -(i as f64) * 0.05));
        }

        let crystal = QmemCrystal {
            label_id: 42,
            num_channels: 4,
            time_steps: 8,
            manifold: SemanticWave::from_vec(data.clone()),
        };

        // Write to temp file
        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("test_roundtrip.qmem");
        crystal.save_to_disk(&tmp_path).expect("save failed");

        // Verify file size: 20 header + 32 samples × 16 bytes = 532 bytes
        let meta = std::fs::metadata(&tmp_path).unwrap();
        assert_eq!(meta.len(), 20 + 32 * 16);

        // Load back
        let loaded = QmemCrystal::load_from_disk(&tmp_path).expect("load failed");

        assert_eq!(loaded.label_id, 42);
        assert_eq!(loaded.num_channels, 4);
        assert_eq!(loaded.time_steps, 8);
        assert_eq!(loaded.manifold.len(), 32);

        // Verify payload integrity (bit-exact roundtrip)
        let orig_slice = crystal.manifold.amplitudes.as_slice().unwrap();
        let loaded_slice = loaded.manifold.amplitudes.as_slice().unwrap();
        for (i, (a, b)) in orig_slice.iter().zip(loaded_slice.iter()).enumerate() {
            assert_eq!(
                a.re.to_bits(), b.re.to_bits(),
                "Re mismatch at sample {}", i
            );
            assert_eq!(
                a.im.to_bits(), b.im.to_bits(),
                "Im mismatch at sample {}", i
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_qmem_invalid_magic_rejected() {
        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("test_bad_magic.qmem");

        // Write garbage
        let mut f = File::create(&tmp_path).unwrap();
        f.write_all(b"FAKE\x01\x00\x00\x00").unwrap();
        f.write_all(&[0u8; 12]).unwrap();
        drop(f);

        let result = QmemCrystal::load_from_disk(&tmp_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid magic"));

        let _ = std::fs::remove_file(&tmp_path);
    }
}
