//! Minimal global allocator shim that delegates to jemalloc and
//! provides graceful memory-pressure signalling instead of abort().
//!
//! ARCHITECTURE:
//!   - At 85% of the configured memory limit, a soft signal is sent via a
//!     registered std::sync::mpsc::SyncSender<()>.
//!   - The system NEVER calls abort() III a long-running enterprise job must
//!     NEVER be killed by a 1-byte limit overshoot.
//!   - Above 85%: signal is sent once per check interval.
//!   - Above 98%: an additional warning is logged so operators can act.
//!
//! Usage (in main.rs):
//!   let (tx, rx) = std::sync::mpsc::sync_channel(1);
//!   sodals_alloc::register_pressure_hook(tx);
//!   // Spawn a listener on `rx` that reduces CortexActor load.

#[cfg(all(not(target_os = "windows"), feature = "jemalloc"))]
use jemallocator::Jemalloc;
#[cfg(all(not(target_os = "windows"), feature = "jemalloc"))]
use jemalloc_ctl::{epoch, stats};

use std::sync::{OnceLock, mpsc::SyncSender};

// make jemalloc the real allocator when explicitly enabled on Unix
#[cfg(all(not(target_os = "windows"), feature = "jemalloc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

// IIIIII Memory ceiling ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static MEMORY_LIMIT: OnceLock<usize> = OnceLock::new();

fn get_memory_limit() -> usize {
    *MEMORY_LIMIT.get_or_init(|| {
        std::env::var("SODALS_MEMORY_LIMIT_GB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|gb| gb * 1024 * 1024 * 1024)
            .unwrap_or(28 * 1024 * 1024 * 1024)
    })
}

// IIIIII Pressure hook III registered once by main.rs ////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIII
static PRESSURE_SENDER: OnceLock<SyncSender<()>> = OnceLock::new();

/// Register a channel that receives a `()` signal when memory is above 85%.
/// Call this once at startup, before starting the monitor.
///
/// The receiver side should live in `CortexActor` (or similar) to reduce load:
/// e.g. drop a pending request, reduce the Ouroboros loop count, or prune the
/// Z-Matrix LRU tail.
pub fn register_pressure_hook(tx: SyncSender<()>) {
    let _ = PRESSURE_SENDER.set(tx);
}

// IIIIII Monitor thread ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Spawn the memory monitor thread. Call this once early in main.rs.
#[cfg(all(not(target_os = "windows"), feature = "jemalloc"))]
pub fn init_memory_monitor() {
    std::thread::Builder::new()
        .name("mem-monitor".into())
        .spawn(|| {
            let limit = get_memory_limit();
            let soft_threshold   = (limit as f64 * 0.85) as usize; // 85% III signal
            let hard_threshold   = (limit as f64 * 0.98) as usize; // 98% III loud warning
            let mut last_signalled = false;

            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));

                let _ = epoch::advance();
                let used = stats::allocated::read().unwrap_or(0);

                if used > hard_threshold {
                    eprintln!(
                        "[MEM] IIIIII  CRITICAL memory pressure: {:.1} GB used / {:.1} GB limit ({:.0}%). \
                         System slowing down but NOT aborting.",
                        used         as f64 / 1_073_741_824.0,
                        limit        as f64 / 1_073_741_824.0,
                        used as f64 / limit as f64 * 100.0,
                    );
                }

                if used > soft_threshold {
                    if !last_signalled {
                        eprintln!(
                            "[MEM] Memory pressure signal sent ({:.1} GB / {:.1} GB III {:.0}% of limit).",
                            used  as f64 / 1_073_741_824.0,
                            limit as f64 / 1_073_741_824.0,
                            used as f64 / limit as f64 * 100.0,
                        );
                        if let Some(tx) = PRESSURE_SENDER.get() {
                            let _ = tx.try_send(());
                        }
                        last_signalled = true;
                    }
                } else {
                    last_signalled = false;
                }
            }
        })
        .expect("failed to start memory monitor");
}

#[cfg(not(all(not(target_os = "windows"), feature = "jemalloc")))]
pub fn init_memory_monitor() {
    std::thread::Builder::new()
        .name("mem-monitor-stub".into())
        .spawn(|| {
            eprintln!("[MEM] Memory monitor stub active (enable feature `jemalloc` on Unix for live stats).");
        })
        .expect("failed to start stub memory monitor");
}

// IIIIII Public helpers ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns current jemalloc allocated bytes (or 0 when jemalloc feature is disabled).
#[cfg(all(not(target_os = "windows"), feature = "jemalloc"))]
pub fn current_usage() -> usize {
    let _ = epoch::advance();
    stats::allocated::read().unwrap_or(0)
}

#[cfg(not(all(not(target_os = "windows"), feature = "jemalloc")))]
pub fn current_usage() -> usize { 0 }

pub fn memory_limit_human() -> String {
    let bytes = get_memory_limit();
    if bytes > 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Returns the configured hard memory ceiling in bytes.
///
/// This is a lightweight wrapper around the internal `get_memory_limit()` and
/// is safe to call from monitoring/telemetry code that needs the numeric value.
pub fn memory_limit_bytes() -> usize {
    get_memory_limit()
}













