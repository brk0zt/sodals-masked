use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, Chart, Dataset, Axis, Paragraph},
    style::{Style, Color, Modifier},
    Terminal,
};
use std::io;

// Phase 25: Modern ratatui API - Frame no longer requires generic parameter
pub fn draw_ui_frame(
    f: &mut ratatui::Frame,
    wave_data: &[(f64, f64)], // Osiloskop verisi (Zaman, Genlik)
    llm_text: &str,           // Qwen çıktısı
    memory_status: &str       // ".qmem kaydedildi" vb.
) {
    // 1. Ekranı Böl (Layout)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(f.size());

    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)].as_ref())
        .split(chunks[0]);

    // 2. SOL ÜST: OSİLOSKOP (Dalga Görünümü)
    let datasets = vec![
        Dataset::default()
            .name("Düşünce Dalgası (Reel)")
            .marker(ratatui::symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .data(wave_data),
    ];
    
    let chart = Chart::new(datasets)
        .block(Block::default().title("SODALS Rezonans Çekirdeği").borders(Borders::ALL))
        .x_axis(Axis::default().title("Zaman (t)").bounds([0.0, 100.0]))
        .y_axis(Axis::default().title("Genlik").bounds([-2.0, 2.0]));
        
    f.render_widget(chart, top_chunks[0]);

    // 3. SAĞ ÜST: SİSTEM BİLGİSİ
    let sys_info = Paragraph::new(format!("CPU: i9-Core Active\nRAM: 64GB OK\nQwen: 4-bit Quantized\n\n{}", memory_status))
        .block(Block::default().title("Sistem").borders(Borders::ALL))
        .style(Style::default().fg(Color::Green));
    f.render_widget(sys_info, top_chunks[1]);

    // 4. ALT KISIM: LLM ÇIKTISI
    let log_box = Paragraph::new(llm_text)
        .block(Block::default().title("Qwen 2.5 İletişim Hattı").borders(Borders::ALL))
        .wrap(ratatui::widgets::Wrap { trim: true });
    f.render_widget(log_box, chunks[1]);
}