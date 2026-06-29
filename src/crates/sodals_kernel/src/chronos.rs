use std::time::{Duration, Instant};

pub struct TimeKeeper {
    start_time: Instant,
    total_budget: Duration,      // 9 hours total
    total_tasks: usize,          // 100 tasks
    tasks_processed: usize,
    
    // Adaptive Fields
    current_task_start: Option<Instant>,
    banked_time: Duration,       // Accumulated savings
}

impl TimeKeeper {
    pub fn new(total_minutes: u64, total_tasks: usize) -> Self {
        Self {
            start_time: Instant::now(),
            total_budget: Duration::from_secs(total_minutes * 60),
            total_tasks,
            tasks_processed: 0,
            current_task_start: None,
            banked_time: Duration::ZERO,
        }
    }

    /// Call when a task starts
    pub fn start_task(&mut self) {
        self.current_task_start = Some(Instant::now());
    }

    /// Call when a task finishes successfully
    pub fn end_task(&mut self) {
        if let Some(start) = self.current_task_start {
            let elapsed = start.elapsed();
            self.tasks_processed += 1;
            
            // Calculate what the "average" budget was for this specific task slot
            // Simple approximation: Total Budget / 100 = 5.4 min
            let nominal_budget = self.total_budget / (self.total_tasks as u32);
            
            if elapsed < nominal_budget {
                // We saved time! Bank it.
                let savings = nominal_budget - elapsed;
                self.banked_time += savings;
                println!("[CHRONOS] 💰 Time Banked: +{:.1}s (Total Bank: {:.1}s)", 
                         savings.as_secs_f64(), self.banked_time.as_secs_f64());
            } else {
                // We overspent. Deduct from bank (or go into debt, logic handles debt via 0 bank)
                let overspend = elapsed - nominal_budget;
                if self.banked_time > overspend {
                    self.banked_time -= overspend;
                } else {
                    self.banked_time = Duration::ZERO;
                }
                println!("[CHRONOS] 📉 Time Overspent: -{:.1}s", overspend.as_secs_f64());
            }
        }
        self.current_task_start = None;
    }

    /// dynamic_budget = Base Share + (Bank * 0.5)
    pub fn get_current_task_budget(&self) -> Duration {
        let elapsed_global = self.start_time.elapsed();
        if elapsed_global >= self.total_budget {
            return Duration::from_secs(1); // Game over logic
        }

        let remaining_global = self.total_budget - elapsed_global;
        let remaining_tasks = (self.total_tasks - self.tasks_processed).max(1) as u32;

        // 1. Base Share: If we have 100 mins and 10 tasks, base is 10 mins.
        let base_share = remaining_global / remaining_tasks;

        // 2. Spending Power: We can use the Base Share + 50% of our Savings
        // We keep 50% of savings for future disasters.
        base_share + (self.banked_time / 2)
    }

    /// Returns true if we should PANIC and stop thinking
    pub fn should_abort(&self) -> bool {
        if let Some(start) = self.current_task_start {
            let limit = self.get_current_task_budget();
            // Abort if we exceed the dynamic limit
            start.elapsed() > limit
        } else {
            false
        }
    }

    /// Returns (Universes, Retries) based on wealth
    pub fn get_complexity_allowance(&self) -> (usize, usize) {
        let budget = self.get_current_task_budget();
        
        // If we have > 5 mins for this task, go full power
        if budget.as_secs() > 300 {
            (20, 3) // ULTRA Mode: 20 Universes!
        } else if budget.as_secs() > 120 {
            (10, 2) // Standard Mode
        } else {
            (5, 1)  // Eco Mode
        }
    }

    /// Get current banked time (for monitoring)
    pub fn get_banked_time(&self) -> Duration {
        self.banked_time
    }

    /// Get tasks remaining
    pub fn get_remaining_tasks(&self) -> usize {
        (self.total_tasks - self.tasks_processed).max(0)
    }

    /// Get global time remaining
    pub fn get_remaining_time(&self) -> Duration {
        let elapsed = self.start_time.elapsed();
        if elapsed >= self.total_budget {
            Duration::ZERO
        } else {
            self.total_budget - elapsed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_time_banking_basic() {
        let mut keeper = TimeKeeper::new(100, 10);
        
        // First task takes 2 minutes, out of a 10-minute budget
        keeper.start_task();
        thread::sleep(Duration::from_secs(2));
        keeper.end_task();
        
        // Should have banked 8 minutes
        assert!(keeper.get_banked_time().as_secs() >= 8);
    }

    #[test]
    fn test_complexity_allowance_ultra_mode() {
        let keeper = TimeKeeper::new(100, 5);
        let (universes, retries) = keeper.get_complexity_allowance();
        
        // With 100 mins and 5 tasks, base is 20 mins per task
        // Should be ULTRA mode
        assert_eq!(universes, 20);
        assert_eq!(retries, 3);
    }

    #[test]
    fn test_should_abort_when_budget_exceeded() {
        let mut keeper = TimeKeeper::new(10, 10);
        keeper.start_task();
        
        // Sleep for 1 full second
        thread::sleep(Duration::from_secs(1));
        
        // With 10 mins budget and 10 tasks, each gets 1 minute
        // After 1 second, we shouldn't abort yet
        assert!(!keeper.should_abort());
    }
}
