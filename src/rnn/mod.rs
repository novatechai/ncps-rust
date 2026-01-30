//! RNN Layer Modules
//!
//! Full RNN layers that handle sequence processing, batching, and state management.
//! These wrap the individual cell implementations to provide complete RNN functionality.

pub mod cfc;
pub mod ltc;

pub use cfc::CfC;
pub use ltc::LTC;
