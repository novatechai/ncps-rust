//! # NCPS - Neural Circuit Policies for Rust
//!
//! [![Crates.io](https://img.shields.io/crates/v/ncps.svg)](https://crates.io/crates/ncps)
//! [![Documentation](https://docs.rs/ncps/badge.svg)](https://docs.rs/ncps)
//!
//! A Rust implementation of Neural Circuit Policies (NCPs) using the [Burn](https://burn.dev)
//! deep learning framework. NCPs are biologically-inspired recurrent neural networks with
//! sparse, structured connectivity patterns.
//!
//! ## Why NCPs?
//!
//! | Feature | Traditional RNN | NCP |
//! |---------|-----------------|-----|
//! | Connectivity | Dense (O(N²)) | Sparse (O(N)) |
//! | Parameters | Many | Few |
//! | Interpretability | Black box | Structured layers |
//! | Time handling | Discrete | Continuous |
//!
//! NCPs are particularly well-suited for:
//! - **Time series** with irregular sampling
//! - **Robotics** and control systems
//! - **Edge deployment** where parameters matter
//! - **Interpretable AI** applications
//!
//! ## Quick Start
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! ncps = "0.1"
//! burn = { version = "0.16", features = ["ndarray"] }
//! ```
//!
//! ### Basic Usage
//!
//! ```rust
//! use ncps::prelude::*;
//!
//! // 1. Create wiring (defines network structure)
//! let mut wiring = AutoNCP::new(
//!     32,    // total neurons
//!     8,     // output size
//!     0.5,   // sparsity (50% connections removed)
//!     42,    // random seed
//! );
//!
//! // 2. Build with your input dimension
//! wiring.build(16);  // 16 input features
//!
//! // 3. Verify configuration
//! assert_eq!(wiring.units(), 32);
//! assert_eq!(wiring.output_dim(), Some(8));
//! assert!(wiring.is_built());
//! ```
//!
//! ### Full RNN Example
//!
//! ```ignore
//! use ncps::prelude::*;
//! use burn::tensor::Tensor;
//! use burn::backend::NdArray;
//!
//! type Backend = NdArray<f32>;
//!
//! let device = Default::default();
//!
//! // Create wiring
//! let mut wiring = AutoNCP::new(64, 10, 0.5, 42);
//! wiring.build(20);  // 20 input features
//!
//! // Create CfC RNN layer
//! let cfc = CfC::<Backend>::with_wiring(20, wiring, &device);
//!
//! // Process sequence: [batch=4, seq_len=100, features=20]
//! let input: Tensor<Backend, 3> = Tensor::zeros([4, 100, 20], &device);
//! let (output, final_state) = cfc.forward(input, None, None);
//!
//! // output: [4, 100, 10] - output at each timestep
//! // final_state: [4, 64] - hidden state for continuation
//! ```
//!
//! ## Architecture Overview
//!
//! ```text
//!                          ┌─────────────────────────┐
//!                          │    Your Application     │
//!                          └───────────┬─────────────┘
//!                                      │
//!                          ┌───────────▼─────────────┐
//!                          │   RNN Layers (cfc, ltc) │  ◄── Use these!
//!                          │   Sequence processing   │
//!                          └───────────┬─────────────┘
//!                                      │
//!                          ┌───────────▼─────────────┐
//!                          │   Cells (cfc_cell, etc) │  ◄── Single timestep
//!                          │   Low-level operations  │
//!                          └───────────┬─────────────┘
//!                                      │
//!                          ┌───────────▼─────────────┐
//!                          │   Wirings (ncp, etc)    │  ◄── Network structure
//!                          │   Connectivity patterns │
//!                          └─────────────────────────┘
//! ```
//!
//! ## Module Guide
//!
//! | Module | Purpose | Start Here? |
//! |--------|---------|-------------|
//! | [`rnn`] | Full RNN layers for sequences | ✅ Yes |
//! | [`wirings`] | Network connectivity patterns | ✅ Yes |
//! | [`cells`] | Single-timestep processing | For advanced use |
//! | [`activation`] | Activation functions | Rarely needed |
//!
//! ## Choosing Components
//!
//! ### RNN Layer: CfC vs LTC
//!
//! | Choose [`CfC`](rnn::CfC) when... | Choose [`LTC`](rnn::LTC) when... |
//! |----------------------------------|----------------------------------|
//! | Speed matters | Biological accuracy matters |
//! | Training large models | Research applications |
//! | Production deployment | Variable time constants needed |
//!
//! ### Wiring: AutoNCP vs NCP vs FullyConnected
//!
//! | Choose [`AutoNCP`](wirings::AutoNCP) when... | Choose [`NCP`](wirings::NCP) when... | Choose [`FullyConnected`](wirings::FullyConnected) when... |
//! |-----------------------------------------------|--------------------------------------|-------------------------------------------------------------|
//! | Starting out (recommended) | Need exact layer sizes | Need baseline comparison |
//! | Want automatic configuration | Fine-tuning connectivity | Maximum expressiveness |
//! | Most use cases | Research/ablation studies | Don't care about sparsity |
//!
//! ## Common Patterns
//!
//! ### Sequence Classification
//!
//! ```ignore
//! // Only return the final output
//! let cfc = CfC::<Backend>::with_wiring(input_size, wiring, &device)
//!     .with_return_sequences(false);
//!
//! let (output, _) = cfc.forward(input, None, None);
//! // output: [batch, 1, output_size]
//! ```
//!
//! ### Stateful Processing (streaming)
//!
//! ```ignore
//! // Preserve state across batches
//! let (out1, state) = cfc.forward(batch1, None, None);
//! let (out2, state) = cfc.forward(batch2, Some(state), None);
//! let (out3, state) = cfc.forward(batch3, Some(state), None);
//! ```
//!
//! ### Different Input Formats
//!
//! ```ignore
//! // Sequence-first format: [seq_len, batch, features]
//! let cfc = CfC::<Backend>::new(input_size, hidden_size, &device)
//!     .with_batch_first(false);
//! ```
//!
//! ## Important: The `.build()` Step
//!
//! **All wirings must be built before use:**
//!
//! ```rust
//! use ncps::wirings::{AutoNCP, Wiring};
//!
//! let mut wiring = AutoNCP::new(32, 8, 0.5, 42);
//!
//! // ❌ Not ready yet - sensory connections don't exist
//! assert!(!wiring.is_built());
//!
//! // ✅ Build with input dimension
//! wiring.build(16);
//! assert!(wiring.is_built());
//!
//! // Now you can use it with RNN layers
//! ```
//!
//! ## References
//!
//! - [Neural Circuit Policies Paper](https://publik.tuwien.ac.at/files/publik_292280.pdf)
//! - [Closed-form Continuous-time Paper](https://arxiv.org/abs/2106.13898)
//! - [Original Python Implementation](https://github.com/mlech26l/ncps)
//!
//! ## Feature Flags
//!
//! This crate uses Burn's backend system. Configure via Burn's features:
//!
//! ```toml
//! # CPU (default)
//! burn = { version = "0.16", features = ["ndarray"] }
//!
//! # GPU (WGPU)
//! burn = { version = "0.16", features = ["wgpu"] }
//!
//! # GPU (Candle)
//! burn = { version = "0.16", features = ["candle"] }
//! ```

pub mod activation;
pub mod cells;
pub mod rnn;
pub mod wirings;

pub mod prelude {
    pub use crate::activation::LeCun;
    pub use crate::cells::{CfCCell, CfcMode, LSTMCell, LTCCell, MappingMode};
    pub use crate::rnn::{CfC, LTC};
    pub use crate::wirings::{AutoNCP, FullyConnected, NCP, Random, Wiring};
}
