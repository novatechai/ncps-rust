//! # RNN Cell Implementations
//!
//! This module provides single-timestep RNN cells for Neural Circuit Policies.
//! These cells process one timestep at a time and are wrapped by the higher-level
//! RNN layers in [`crate::rnn`] for sequence processing.
//!
//! ## Cell Types
//!
//! | Cell | Description | Use Case |
//! |------|-------------|----------|
//! | [`CfCCell`] | Closed-form Continuous-time | Fast, efficient, **recommended** |
//! | [`LTCCell`] | Liquid Time-Constant | Biologically accurate, slower |
//! | [`WiredCfCCell`] | CfC with multi-layer wiring | Complex architectures |
//! | [`LSTMCell`] | Standard LSTM | Mixed memory augmentation |
//!
//! ## When to Use Cells Directly
//!
//! Most users should use the higher-level [`CfC`](crate::rnn::CfC) or [`LTC`](crate::rnn::LTC)
//! layers which handle sequence processing automatically. Use cells directly when you need:
//!
//! - Custom sequence processing logic
//! - Integration with other frameworks
//! - Fine-grained control over state management
//!
//! ## CfC Operating Modes
//!
//! The [`CfCCell`] supports three operating modes via [`CfcMode`]:
//!
//! ### Default Mode (Recommended)
//! ```text
//! h = tanh(ff1) × (1 - σ(t)) + tanh(ff2) × σ(t)
//! ```
//! Gated interpolation between two feedforward paths. Best balance of
//! expressiveness and stability.
//!
//! ### Pure Mode
//! ```text
//! h = a - a × exp(-t × (|w_τ| + |ff1|)) × ff1
//! ```
//! Direct ODE solution without gating. More biologically plausible but
//! can be less stable for some tasks.
//!
//! ### NoGate Mode
//! ```text
//! h = tanh(ff1) + tanh(ff2) × σ(t)
//! ```
//! Simplified mode using addition instead of interpolation. Useful for
//! tasks where gating adds unnecessary complexity.
//!
//! ## Tensor Shapes
//!
//! All cells expect 2D tensors for single-timestep processing:
//!
//! | Tensor | Shape | Description |
//! |--------|-------|-------------|
//! | `input` | `[batch, input_size]` | Input features |
//! | `hidden_state` | `[batch, hidden_size]` | Previous hidden state |
//! | `output` | `[batch, hidden_size]` | Cell output |
//! | `new_state` | `[batch, hidden_size]` | Updated hidden state |
//!
//! ## Example: Using CfCCell Directly
//!
//! ```ignore
//! use ncps::cells::{CfCCell, CfcMode};
//! use burn::tensor::Tensor;
//!
//! let device = Default::default();
//! let cell = CfCCell::<Backend>::new(16, 32, &device)
//!     .with_mode(CfcMode::Default);
//!
//! // Process single timestep
//! let input: Tensor<Backend, 2> = /* [batch, 16] */;
//! let hidden: Tensor<Backend, 2> = Tensor::zeros([batch, 32], &device);
//!
//! let (output, new_hidden) = cell.forward(input, hidden, 1.0);
//! // output: [batch, 32]
//! // new_hidden: [batch, 32]
//! ```
//!
//! ## Input/Output Mapping Modes
//!
//! [`LTCCell`] supports different input/output mapping strategies via [`MappingMode`]:
//!
//! - **Affine**: Linear transformation with bias (most expressive)
//! - **Linear**: Linear transformation without bias
//! - **None**: Direct pass-through (fastest)

pub mod cfc_cell;
pub mod lstm_cell;
pub mod ltc_cell;
pub mod wired_cfc_cell;

pub use cfc_cell::{CfCCell, CfcMode};
pub use lstm_cell::LSTMCell;
pub use ltc_cell::{LTCCell, MappingMode};
pub use wired_cfc_cell::WiredCfCCell;
