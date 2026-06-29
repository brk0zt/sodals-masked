// Structured logging utilities shared by multiple crates

/// Log level enum for backward compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    SYSTEM,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::DEBUG => "DEBUG",
            LogLevel::INFO => "INFO",
            LogLevel::WARN => "WARN",
            LogLevel::ERROR => "ERROR",
            LogLevel::SYSTEM => "SYSTEM",
        }
    }
}

/// Legacy log manager for backward compatibility
///
/// In new code, prefer using `tracing` macros directly:
/// ```rust
/// use tracing::{info, warn, error, debug};
///
/// info!("Server started");
/// warn!("High memory usage");
/// error!("Connection failed");
/// ```
#[derive(Clone)]
pub struct LogManager {
    log_level: LogLevel,
}

impl LogManager {
    /// Static logging (uses tracing under the hood)
    pub fn log(level: LogLevel, source: &str, message: &str) {
        eprintln!("[{}] [{}] {}", level.as_str(), source, message);
    }

    /// Create new log manager
    pub fn new(log_level: LogLevel) -> Self {
        Self { log_level }
    }
    
    /// Legacy: Create with TUI channel (ignored in modern version)
    #[deprecated(note = "Use tracing directly instead")]
    pub fn with_tui(_tui_tx: tokio::sync::mpsc::Sender<String>, log_level: LogLevel) -> Self {
        Self::new(log_level)
    }
    
    /// Check if a log level should be logged
    fn should_log(&self, level: LogLevel) -> bool {
        let priority = |l: LogLevel| match l {
            LogLevel::DEBUG => 0,
            LogLevel::INFO => 1,
            LogLevel::WARN => 2,
            LogLevel::ERROR => 3,
            LogLevel::SYSTEM => 4,
        };
        
        priority(level) >= priority(self.log_level)
    }
    
    /// Log with level filtering
    fn log_filtered(&self, level: LogLevel, source: &str, message: &str) {
        if self.should_log(level) {
            Self::log(level, source, message);
        }
    }
    
    /// Convenience methods
    pub fn debug(&self, source: &str, message: &str) {
        self.log_filtered(LogLevel::DEBUG, source, message);
    }
    
    pub fn info(&self, source: &str, message: &str) {
        self.log_filtered(LogLevel::INFO, source, message);
    }
    
    pub fn warn(&self, source: &str, message: &str) {
        self.log_filtered(LogLevel::WARN, source, message);
    }
    
    pub fn error(&self, source: &str, message: &str) {
        self.log_filtered(LogLevel::ERROR, source, message);
    }
    
    pub fn system(&self, source: &str, message: &str) {
        self.log_filtered(LogLevel::SYSTEM, source, message);
    }
}

impl Default for LogManager {
    fn default() -> Self {
        Self::new(LogLevel::INFO)
    }
}

/// Legacy logging macro (deprecated - use tracing::info! instead)
#[macro_export]
macro_rules! log_info {
    ($source:expr, $fmt:literal $(, $args:expr )* $(,)?) => {{
        $crate::log_manager::LogManager::log(
            $crate::log_manager::LogLevel::INFO,
            $source,
            &format!($fmt $(, $args )*)
        );
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_log() {
        let logger = LogManager::new(LogLevel::WARN);
        
        assert!(!logger.should_log(LogLevel::DEBUG));
        assert!(!logger.should_log(LogLevel::INFO));
        assert!(logger.should_log(LogLevel::WARN));
        assert!(logger.should_log(LogLevel::ERROR));
        assert!(logger.should_log(LogLevel::SYSTEM));
    }
}
