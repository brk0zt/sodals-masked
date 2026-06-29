// src/job_manager.rs - Job Tracking and Timeout Management

use std::collections::HashMap;
use std::time::{Instant, Duration};

#[derive(Debug, Clone)]
pub struct Job {
    pub req_id: u64,
    pub user_id: String,
    pub prompt: String,
    pub created_at: Instant,
    pub timeout: Duration,
}

pub struct JobManager {
    jobs: HashMap<u64, Job>,
    default_timeout: Duration,
}

impl JobManager {
    pub fn new(default_timeout_seconds: u64) -> Self {
        Self {
            jobs: HashMap::new(),
            default_timeout: Duration::from_secs(default_timeout_seconds),
        }
    }
    
    /// Add a new job to the manager
    pub fn add_job(&mut self, req_id: u64, user_id: String, prompt: String, timeout_secs: Option<u64>) {
        let timeout = timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(self.default_timeout);
        
        let job = Job {
            req_id,
            user_id,
            prompt,
            created_at: Instant::now(),
            timeout,
        };
        
        self.jobs.insert(req_id, job);
    }
    
    /// Get a job by request ID
    pub fn get_job(&self, req_id: &u64) -> Option<&Job> {
        self.jobs.get(req_id)
    }
    
    /// Remove a job (when completed)
    pub fn remove_job(&mut self, req_id: &u64) -> Option<Job> {
        self.jobs.remove(req_id)
    }
    
    /// Check for timed-out jobs and return their IDs
    pub fn check_timeouts(&self) -> Vec<u64> {
        let now = Instant::now();
        let mut timed_out = Vec::new();
        
        for (req_id, job) in &self.jobs {
            let elapsed = now.duration_since(job.created_at);
            if elapsed > job.timeout {
                timed_out.push(*req_id);
            }
        }
        
        timed_out
    }
    
    /// Get all jobs (for debugging/admin)
    pub fn get_all_jobs(&self) -> &HashMap<u64, Job> {
        &self.jobs
    }
    
    /// Clean up timed-out jobs
    pub fn cleanup_timeouts(&mut self) -> Vec<Job> {
        let timed_out_ids = self.check_timeouts();
        let mut removed = Vec::new();
        
        for req_id in timed_out_ids {
            if let Some(job) = self.jobs.remove(&req_id) {
                removed.push(job);
            }
        }
        
        removed
    }
}
