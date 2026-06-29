// src/crates/sodals_kernel/src/main.rs

use tokio::sync::{mpsc, broadcast, RwLock};
use std::sync::Arc;
use anyhow::Result;
// YENİ: Gerekli importlar
use tokio::time::{interval, Duration};
use std::io::Write;

// External crates
use sodals_common::config::SystemConfig;
use sodals_net::socket_actor::SocketActor;
use tracing::{info, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use std::collections::HashMap;
// Neuro package
use sodals_net::cortex_actor::CortexActor;
use sodals_neuro::semantic_network::ResonanceNetwork;
use sodals_neuro::sodals_alloc; // allocator monitor

use sodals_kernel::kernel::{Kernel, BrainHandle, brain_actor}; 
// use sodals_kernel::forge::ForgeActor;
// use sodals_kernel::runtime::RuntimeActor;
// use sodals_kernel::store::VectorStoreActor;
// use sodals_kernel::sentinel::SentinelActor;
// use sodals_kernel::job_manager::JobManager;
// Not: 'mod' satırlarını siliyoruz çünkü lib.rs'de tanımladık.


/// Initialize tracing with reasonable defaults
fn init_logging() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,sodals=debug".into())
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {

    dotenvy::dotenv().ok();
    // 1. Logging
    init_logging();
    info!("SODALS: Starting unified architecture");

    // early allocator monitor
    sodals_alloc::init_memory_monitor();

    // 2. Configuration
    let config = SystemConfig::load()
        .map_err(|e| anyhow::anyhow!("Config yüklenemedi: {}", e))?;

    // 3. Shared brain actor – lock‑free snapshot service
    let (brain_tx, brain_rx) = mpsc::channel(32);
    let initial_brain = ResonanceNetwork::new(10000);
    let brain_handle = BrainHandle::new(brain_tx);
    tokio::spawn(brain_actor(brain_rx, initial_brain));
    let brain = Arc::new(brain_handle);

    // 4. Channel strategy:
    let (tui_tx, tui_rx) = mpsc::channel(32);  // TUI events
    let (cortex_tx, cortex_rx) = mpsc::channel(512); // AI inference
    let (kernel_tx, kernel_rx) = mpsc::channel(1024); // Kernel main loop
    let (broadcast_tx, _broadcast_rx) = broadcast::channel(100); // System broadcast

    // 5. Spawn actors
    
    // NOT: Claude'un önerdiği loop TUI görevi göreceği için TuiActor'ü devre dışı bırakıyoruz.
    // Eğer TuiActor (Ratatui) kullanmak isterseniz burayı açıp aşağıdaki run_main_loop'u kapatmalısınız.
    /*
    let tui_handle = tokio::spawn({
        let tui = TuiActor::new(tui_rx, kernel_tx.clone());
        async move {
            if let Err(e) = tui.run().await {
                error!("TUI actor crashed: {}", e);
            }
        }
    });
    */

    // Cortex: CPU-heavy, spawn on dedicated runtime if needed
    let cortex = CortexActor::new(
        cortex_rx,
        kernel_tx.clone(),
        config.clone(),
        None, // mem_pressure_rx: Option<std::sync::mpsc::Receiver<()>>
    );
    
    let cortex_handle = tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(cortex.run());
    });

    let sessions = Arc::new(RwLock::new(HashMap::new())); // Oturum kayıt defteri

    let _socket_handle = tokio::spawn({
        let k_tx = kernel_tx.clone();
        let port = config.port;
        let secret = config.secret_api_key.clone();
        
        // SocketActor::new(kernel_tx, port, secret_key, sessions) bekliyor
        let socket = SocketActor::new(
            k_tx,
            port,
            secret,
            sessions
        );

        async move {
            info!("WebSocket Server başlatılıyor... (Port: {})", port);
            // Hata yakalama (if let Err) kaldırıldı çünkü run() geriye () dönüyor.
            socket.run().await; 
        }
    });

    // 6. Kernel (main coordinator)
    let kernel = Kernel::new(
        kernel_rx,
        kernel_tx.clone(), // <--- YENİ EKLENEN: Adres kartını teslim ediyoruz
        tui_tx.clone(),
        cortex_tx.clone(),
        broadcast_tx,
        config,
        brain,
    )?; // Kernel::new Result dönebilir, ? ekledik (Kernel koduna göre değişebilir)

    info!("System initialized, entering main loop");

    // 7. Kernel'i Arka Planda Başlat
    // Kernel sonsuz döngüde çalışacağı için onu spawn ediyoruz.
    let _kernel_handle = tokio::spawn(async move {
        if let Err(e) = kernel.run().await {
            error!("Kernel crashed: {}", e);
        }
    });

    // 8. Ana Etkileşim Döngüsünü Başlat (Claude'un Önerisi)
    // TuiActor yerine bu fonksiyon stdin/stdout üzerinden arayüz sağlar.
    // tui_rx: Kernel'dan gelen mesajları okur.
    // kernel_tx: Kernel'a istek gönderir.
    run_main_loop(tui_rx, kernel_tx.clone()).await?;

    // 9. Graceful shutdown
    info!("Shutting down...");
    
    // Kanalları kapat
    drop(cortex_tx);
    drop(kernel_tx);

    // Aktörleri bekle
    let shutdown_timeout = tokio::time::Duration::from_secs(5);
    
    if !cortex_handle.is_finished() {
        tokio::select! {
            _ = cortex_handle => info!("Cortex shutdown complete"),
            _ = tokio::time::sleep(shutdown_timeout) => error!("Cortex shutdown timeout"),
        }
    }

    info!("SODALS shutdown complete");
    Ok(())
}

// ============================================
// CLAUDE SUGGESTION: Robust Main Loop
// ============================================

pub async fn run_main_loop(
    mut kernel_rx: tokio::sync::mpsc::Receiver<sodals_net::protocol::SystemMsg>,
    kernel_tx: tokio::sync::mpsc::Sender<sodals_net::protocol::SystemMsg>,
) -> anyhow::Result<()> {
    use sodals_net::protocol::SystemMsg;
    
    println!("\n╔════════════════════════════════════════╗");
    println!("║  SODALS System Ready                   ║");
    println!("║  Type your message and press Enter     ║");
    println!("║  Type 'quit' to exit                   ║");
    println!("╚════════════════════════════════════════╝\n");
    
    // Spawn a separate thread for user input (non-blocking)
    let (input_tx, mut input_rx) = tokio::sync::mpsc::channel::<String>(10);
    
    tokio::task::spawn_blocking(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        let mut lines = stdin.lock().lines();
        
        while let Some(Ok(line)) = lines.next() {
            if input_tx.blocking_send(line).is_err() {
                break; // Channel closed
            }
        }
    });
    
    // Request counter for unique IDs
    let mut req_counter = 0u64;
    
    // Heartbeat for debugging
    let mut heartbeat = interval(Duration::from_secs(5));
    
    loop {
        tokio::select! {
            // User input from stdin
            Some(user_input) = input_rx.recv() => {
                let trimmed = user_input.trim();
                
                if trimmed.is_empty() {
                    continue;
                }
                
                if trimmed.eq_ignore_ascii_case("quit") || trimmed.eq_ignore_ascii_case("exit") {
                    println!("Shutting down...");
                    let _ = kernel_tx.send(SystemMsg::Shutdown).await;
                    break;
                }
                
                if trimmed.eq_ignore_ascii_case("help") || trimmed == "?" {
                    println!("\nCommands:");
                    println!("  quit/exit - Shutdown system");
                    println!("  help/?    - Show this message");
                    println!("  status    - Show system status");
                    println!("  Any other text - Send to AI\n");
                    continue;
                }
                
                if trimmed.eq_ignore_ascii_case("status") {
                    println!("System Status: Running");
                    println!("Model: Loaded and ready");
                    continue;
                }
                
                // Send to LLM
                req_counter += 1;
                println!("\n[You]: {}", trimmed);
                print!("[AI]: "); // Cevap bekleniyor işareti
                std::io::stdout().flush()?;
                
                let _ = kernel_tx.send(SystemMsg::LlmRequest {
                    req_id: req_counter,
                    user_id: "console".to_string(),
                    prompt: trimmed.to_string(),
                    image_data: None,
                    response_tx: None,
                }).await;
            }
            
            // System messages (from Cortex, etc.)
            Some(msg) = kernel_rx.recv() => {
                match msg {
                    SystemMsg::CortexResponse { content, .. } => {
                        println!("{}\n", content);
                        print!("> ");
                        std::io::stdout().flush()?;
                    }
                    
                    SystemMsg::TokenStream { token } => {
                        print!("{}", token);
                        std::io::stdout().flush()?;
                    }
                    
                    SystemMsg::CortexStatus { message, state } => {
                        println!("\n[STATUS] {}: {}", state, message);
                        print!("> ");
                        std::io::stdout().flush()?;
                    }
                    
                    SystemMsg::Shutdown => {
                        println!("Shutdown signal received");
                        break;
                    }
                    
                    _ => {
                        // Ignore other messages (like internal signaling)
                    }
                }
            }
            
            // Heartbeat (debug only - remove in production)
            _ = heartbeat.tick() => {
                // Sessiz heartbeat, log kirletmemesi için commentlendi
                // eprintln!("[DEBUG] Main loop alive at {}", chrono::Local::now().format("%H:%M:%S"));
            }
        }
    }
    
    println!("Main loop exited");
    Ok(())
}