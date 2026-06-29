// re-export shared utilities from the common crate so other modules
// (and sibling crates) can continue to use the familiar paths.
pub use sodals_common::{config, log_manager};

pub mod protocol;
pub mod wire_protocol; // Eklendi
pub mod socket_actor;  // Eklendi
pub mod web_search;    // Eklendi
pub mod server;        // REST API Gateway
pub mod cortex_actor; // Export CortexActor for other crates