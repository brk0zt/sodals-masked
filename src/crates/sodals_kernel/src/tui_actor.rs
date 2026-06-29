// src/tui_actor.rs - VISUAL POLISH & STATUS MONITOR

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{prelude::*, widgets::*};
use std::{io, time::Duration};
use tokio::sync::mpsc;
use sodals_net::protocol::{SystemMsg, SystemState};

struct AppState {
    logs: Vec<String>,
    gas_history: Vec<u64>,      // CPU/Düşünme Yükü
    sentinel_history: Vec<u64>, // Sentinel Gürültü Seviyesi
    status: SystemState,        // Enum state
    forge_status: String,
    active_jobs: usize,
    iot_status: String,
    tick_count: u64,            // Animasyon için sayaç
    token_stream: String,       // Current token stream for THINKING phase
}

impl AppState {
    fn new() -> Self {
        Self {
            logs: vec!["SODALS CONTROL ROOM: Bağlantı Kuruldu...".into()],
            gas_history: vec![0; 100],
            sentinel_history: vec![0; 100],
            status: SystemState::Booting,
            forge_status: "HAZIR".into(),
            active_jobs: 0,
            iot_status: "OFFLINE".into(),
            tick_count: 0,
            token_stream: String::new(),
        }
    }
}

pub struct TuiActor {
    rx: mpsc::Receiver<SystemMsg>,
    kernel_tx: mpsc::Sender<SystemMsg>,
}

impl TuiActor {
    pub fn new(rx: mpsc::Receiver<SystemMsg>, kernel_tx: mpsc::Sender<SystemMsg>) -> Self {
        Self { rx, kernel_tx }
    }

    pub async fn run(mut self) -> Result<()> {
        // Terminali Hazırla
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut app = AppState::new();
        // Ekran tazeleme hızı (FPS): 100ms (10 FPS) idealdir.
        let mut tick_interval = tokio::time::interval(Duration::from_millis(100));

        loop {
            terminal.draw(|f| ui(f, &app))?;

            tokio::select! {
                // 1. SİSTEM MESAJLARI (Veri Güncelleme)
                Some(msg) = self.rx.recv() => {
                    match msg {
                        SystemMsg::DashboardUpdate {active_jobs, gas_used_last_tick, sentinel_noise, last_log, system_status } => {
                            app.active_jobs = active_jobs;
                            app.status = system_status.clone();
                            
                            // Clear token stream when status changes away from THINKING
                            if system_status != SystemState::Thinking {
                                app.token_stream.clear();
                            }
                            
                            // Grafikleri kaydır (Real-time hissi için)
                            app.gas_history.push(gas_used_last_tick);
                            if app.gas_history.len() > 100 { app.gas_history.remove(0); }
                            
                            app.sentinel_history.push(sentinel_noise);
                            if app.sentinel_history.len() > 100 { app.sentinel_history.remove(0); }

                            if !last_log.is_empty() {
                                let time_str = chrono::Local::now().format("%H:%M:%S");
                                // Truncate very long log messages to prevent layout issues (safe UTF-8)
                                let truncated_log = if last_log.chars().count() > 100 {
                                    format!("{}...", last_log.chars().take(100).collect::<String>())
                                } else {
                                    last_log.clone()
                                };
                                app.logs.push(format!("[{}] {}", time_str, truncated_log));
                                // Log penceresi taşmasın, son 15 satırı tut (reduced for better layout)
                                if app.logs.len() > 15 { app.logs.remove(0); }
                            }
                        }
                        SystemMsg::TokenStream { token } => {
                            // Accumulate locally — no more O(N²) wire transfer
                            app.token_stream.push_str(&token);
                        }
                        SystemMsg::Shutdown => break,
                        _ => {}
                    }
                }
                
                // 2. ZAMANLAYICI (Animasyon ve Klavye)
                _ = tick_interval.tick() => {
                    app.tick_count += 1;
                    
                    // Klavye kontrolü (Non-blocking)
                    if event::poll(Duration::from_millis(0))? {
                        if let Event::Key(key) = event::read()? {
                            if key.code == KeyCode::Char('q') {
                                let _ = self.kernel_tx.send(SystemMsg::Shutdown).await;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Çıkışta Terminali Temizle
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        Ok(())
    }
}

fn ui(f: &mut Frame, app: &AppState) {
    // Adjust layout based on whether we're showing token stream
    let constraints = if app.status == SystemState::Thinking && !app.token_stream.is_empty() {
        vec![
            Constraint::Length(3),  // Header (Durum Çubuğu)
            Constraint::Length(12), // Grafikler (Beyin & Güvenlik)
            Constraint::Length(8),  // Token Stream (when thinking)
            Constraint::Min(10),    // Loglar (Akış)
            Constraint::Length(3),  // Footer (Bilgi)
        ]
    } else {
        vec![
            Constraint::Length(3),  // Header (Durum Çubuğu)
            Constraint::Length(12), // Grafikler (Beyin & Güvenlik)
            Constraint::Min(10),    // Loglar (Akış)
            Constraint::Length(3),  // Footer (Bilgi)
        ]
    };
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(f.size());

    // --- 1. HEADER (DURUM GÖSTERGESİ) ---
    // Duruma göre renk ve yanıp sönme efekti
    let (status_color, status_text) = match &app.status {
        SystemState::Thinking => {
            // Yanıp sönen efekt: "THINKING." -> "THINKING.." -> "THINKING..."
            let dots = ".".repeat((app.tick_count % 4) as usize);
            (Color::Yellow, format!("PROCESSOR BUSY: NEURAL INFERENCE{}", dots))
        },
        SystemState::Searching => (Color::Blue, "MODE: WEB CRAWLER ACTIVE".to_string()),
        SystemState::Secure => (Color::Green, "MODE: SENTINEL SHIELD ACTIVE".to_string()),
        SystemState::Panic(_) => (Color::Red, "SYSTEM ALERT: ERROR STATE".to_string()),
        SystemState::Alarm(_) => (Color::Red, "SYSTEM ALERT: ALARM".to_string()),
        _ => (Color::White, "SYSTEM: IDLE / READY".to_string()),
    };
    
    let header = Paragraph::new(status_text)
        .style(Style::default().fg(status_color).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(status_color)));
    f.render_widget(header, chunks[0]);

    // --- 2. GRAFİKLER ---
    let graph_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Sol: Sentinel (Güvenlik) - Mavi/Cyan
    let sentinel_spark = Sparkline::default()
        .block(Block::default().title(" Sentinel Entropy (Privacy) ").borders(Borders::ALL))
        .data(&app.sentinel_history)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sentinel_spark, graph_chunks[0]);

    // Sağ: Brain Activity (Gas) - Kırmızı/Turuncu
    let gas_spark = Sparkline::default()
        .block(Block::default().title(" Neural Load (Compute) ").borders(Borders::ALL))
        .data(&app.gas_history)
        .style(Style::default().fg(if app.status == SystemState::Thinking { Color::Red } else { Color::Green }));
    f.render_widget(gas_spark, graph_chunks[1]);

    // --- 3. TOKEN STREAM (when THINKING) ---
    let log_index = if app.status == SystemState::Thinking && !app.token_stream.is_empty() {
        // Show token stream - limit display to prevent layout issues
        let stream_text = if app.token_stream.len() > 400 {
            // Show last 400 chars with ellipsis
            format!("...{}", &app.token_stream[app.token_stream.len().saturating_sub(400)..])
        } else {
            app.token_stream.clone()
        };
        
        // Clean up the text for display (remove control characters that might break layout)
        let clean_text = stream_text
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
            .collect::<String>();
        
        let stream_paragraph = Paragraph::new(clean_text)
            .style(Style::default().fg(Color::Yellow))
            .wrap(ratatui::widgets::Wrap { trim: true })
            .block(Block::default()
                .title(" 🧠 Neural Inference Output (Live) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow)));
        f.render_widget(stream_paragraph, chunks[2]);
        3 // Logs are at index 3
    } else {
        2 // Logs are at index 2
    };

    // --- 4. LOGLAR ---
    let logs: Vec<ListItem> = app.logs.iter().rev() // En yeni en üstte
        .take(15) // Limit to 15 most recent logs to prevent overflow
        .map(|s| {
            // Truncate long log lines to prevent layout issues (safe UTF-8)
            let display_text = if s.chars().count() > 80 {
                format!("{}...", s.chars().take(80).collect::<String>())
            } else {
                s.clone()
            };
            
            // Log satırlarını renklendir
            let style = if s.contains("HATA") || s.contains("Error") || s.contains("❌") {
                Style::default().fg(Color::Red)
            } else if s.contains("Sentinel") {
                Style::default().fg(Color::Cyan)
            } else if s.contains("Biyolojik") || s.contains("NEURO") || s.contains("🧠") {
                Style::default().fg(Color::Yellow)
            } else if s.contains("✅") || s.contains("SUCCESS") {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Gray)
            };
            ListItem::new(Line::from(Span::styled(display_text, style)))
        })
        .collect();
        
    let logs_widget = List::new(logs)
        .block(Block::default().title(" System Event Stream ").borders(Borders::ALL))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD));
    f.render_widget(logs_widget, chunks[log_index]);

    // --- 5. FOOTER ---
    let footer_text = format!(
        "FORGE: {} | IoT: {} | Active Threads: {} | PRESS 'q' TO SHUTDOWN", 
        app.forge_status, app.iot_status, app.active_jobs
    );
    
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::TOP));
    let footer_index = if app.status == SystemState::Thinking && !app.token_stream.is_empty() {
        4
    } else {
        3
    };
    f.render_widget(footer, chunks[footer_index]);
}