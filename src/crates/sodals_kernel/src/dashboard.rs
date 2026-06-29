use std::{io, time::Duration};
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{prelude::*, widgets::*};
use tokio::sync::mpsc;
use tokio::time::interval;

// --- VERİ YAPILARI ---

/// SODALS'un diğer modüllerinden TUI'ye gelen olaylar
pub enum SystemEvent {
    Log(String),              // Genel loglar
    SentinelUpdate(u64),      // Gürültü seviyesi (0-100)
    ForgeStatus(String),      // Derleme durumu
    Quit,                     // Çıkış sinyali
}

/// Arayüzün o anki durumunu tutan yapı
struct App {
    logs: Vec<String>,
    sentinel_history: Vec<u64>, // Sparkline grafiği için geçmiş veriler
    forge_status: String,
    cpu_load: f64,
}

impl App {
    fn new() -> Self {
        Self {
            logs: vec![
                "SODALS v0.1 Başlatılıyor...".to_string(),
                "Core: Bellek Hazır".to_string(),
                "LanceDB: Bağlantı Başarılı".to_string(),
            ],
            sentinel_history: vec![0; 50], // Son 50 veri noktası
            forge_status: "BEKLEMEDE".to_string(),
            cpu_load: 0.0,
        }
    }

    fn on_tick(&mut self) {
        // Simülasyon: CPU yükünü hafifçe değiştir
        self.cpu_load = (self.cpu_load + 0.5) % 100.0;
    }
}

// --- ANA ÇALIŞTIRMA FONKSİYONU ---

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Kanal Oluşturma (Backend -> UI iletişimi için)
    let (tx, mut rx) = mpsc::channel::<SystemEvent>(100);

    // 2. BACKEND SİMÜLASYONU (Ayrı Thread)
    // Gerçekte burası senin SODALS Core, Forge ve WebSocket kodun olacak.
    let tx_clone = tx.clone();
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_millis(500));
        let mut rng_tick = 0;
        
        loop {
            interval.tick().await;
            rng_tick += 1;

            // Sentinel Verisi Gönder
            let noise = (rand::random::<u64>() % 40) + 10; // 10-50 arası
            let _ = tx_clone.send(SystemEvent::SentinelUpdate(noise)).await;

            // Arada bir Log at
            if rng_tick % 5 == 0 {
                let _ = tx_clone.send(SystemEvent::Log(format!(
                    "Sentinel: Gürültü matrisi güncellendi [T={}]", rng_tick
                ))).await;
            }

            // Arada bir Forge işlemi simüle et
            if rng_tick % 20 == 0 {
                 let _ = tx_clone.send(SystemEvent::ForgeStatus("DERLENİYOR...".to_string())).await;
                 tokio::time::sleep(Duration::from_millis(1000)).await;
                 let _ = tx_clone.send(SystemEvent::Log("Forge: Yeni modül yüklendi (Hot-Swap)".to_string())).await;
                 let _ = tx_clone.send(SystemEvent::ForgeStatus("HAZIR".to_string())).await;
            }
        }
    });

    // 3. TERMİNAL AYARLARI (Raw Mode)
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // 4. UI DÖNGÜSÜ
    let app = App::new();
    let res = run_app(&mut terminal, app, rx).await;

    // 5. ÇIKIŞ VE TEMİZLİK
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err);
    }

    Ok(())
}

/// Arayüzün çizim ve event döngüsü
async fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
    mut rx: mpsc::Receiver<SystemEvent>,
) -> Result<()> {
    loop {
        // --- ÇİZİM ---
        terminal.draw(|f| ui(f, &mut app))?;

        // --- OLAY YÖNETİMİ ---
        // Burada hem klavye girişlerini hem de kanaldan gelen veriyi bekliyoruz.
        // `tokio::select!` makrosu, hangisi önce gelirse onu çalıştırır.
        
        tokio::select! {
            // 1. Kanal Verisi (Backend'den gelenler)
            Some(event) = rx.recv() => {
                match event {
                    SystemEvent::Log(msg) => {
                        app.logs.push(msg);
                        if app.logs.len() > 20 { app.logs.remove(0); } // Son 20 log
                    }
                    SystemEvent::SentinelUpdate(val) => {
                        app.sentinel_history.push(val);
                        if app.sentinel_history.len() > 100 { app.sentinel_history.remove(0); }
                    }
                    SystemEvent::ForgeStatus(status) => {
                        app.forge_status = status;
                    }
                    SystemEvent::Quit => return Ok(()),
                }
            }
            
            // 2. Klavye Girişi (Kullanıcıdan gelenler)
            // Crossterm eventleri bloklayan bir işlemdir, bu yüzden ayrı bir async task ile sarmalamak
            // veya non-blocking poll yapmak gerekir. Basitlik için burada poll kullanacağız.
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if event::poll(Duration::from_millis(0))? {
                    if let Event::Key(key) = event::read()? {
                        if key.code == KeyCode::Char('q') {
                            return Ok(());
                        }
                    }
                }
                app.on_tick();
            }
        }
    }
}

/// Arayüzün Görsel Tasarımı (Widget Yerleşimi)
fn ui(f: &mut Frame, app: &mut App) {
    // Ekranı dikey olarak 3'e böl: [Başlık, Orta Kısım, Alt Bilgi]
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Başlık
            Constraint::Min(10),    // Ana içerik
            Constraint::Length(3),  // Alt bilgi
        ])
        .split(f.size());

    // --- 1. BAŞLIK ---
    let title = Paragraph::new("SODALS OPERATING SYSTEM - SENTINEL ACTIVE")
        .style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // --- 2. ORTA KISIM (Yatay Bölme) ---
    // [Sentinel Grafiği (%60) | Loglar (%40)]
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Sol: Sentinel Monitor (Sparkline)
    let sparkline = Sparkline::default()
        .block(Block::default().title(" Sentinel Noise Level (Wronskian Metric) ").borders(Borders::ALL))
        .data(&app.sentinel_history)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, main_chunks[0]);

    // Sağ: Sistem Logları (List)
    let logs: Vec<ListItem> = app
        .logs
        .iter()
        .map(|m| ListItem::new(Line::from(vec![Span::raw(m)])))
        .collect();
    let logs_widget = List::new(logs)
        .block(Block::default().title(" System Logs ").borders(Borders::ALL))
        .style(Style::default().fg(Color::White));
    f.render_widget(logs_widget, main_chunks[1]);

    // --- 3. ALT BİLGİ (Forge Durumu ve Çıkış) ---
    let footer_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[2]);

    let forge_info = Paragraph::new(format!("FORGE STATUS: {}", app.forge_status))
        .style(Style::default().fg(if app.forge_status == "HAZIR" { Color::Green } else { Color::Red }))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(forge_info, footer_chunks[0]);

    let help_info = Paragraph::new("Press 'q' to Shutdown")
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Right)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(help_info, footer_chunks[1]);
}