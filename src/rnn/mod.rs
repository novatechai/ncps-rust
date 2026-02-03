//! # RNN Layers for Sequence Processing
//!
//! This module provides complete RNN layers that handle sequence processing,
//! batching, and hidden state management. **These are the primary APIs most users should use.**
//!
//! ## Available Layers
//!
//! | Layer | Description | Speed | Biological Accuracy |
//! |-------|-------------|-------|---------------------|
//! | [`CfC`] | Closed-form Continuous-time RNN | ⚡ Fast | Medium |
//! | [`LTC`] | Liquid Time-Constant RNN | 🐢 Slower | High |
//!
//! ## Quick Start
//!
//! ```ignore
//! use ncps::prelude::*;
//! use burn::tensor::Tensor;
//!
//! // Create CfC layer with wiring
//! let mut wiring = AutoNCP::new(32, 8, 0.5, 42);
//! wiring.build(16);
//!
//! let cfc = CfC::<Backend>::with_wiring(16, wiring, &device);
//!
//! // Process sequence: [batch=4, seq_len=10, features=16]
//! let input: Tensor<Backend, 3> = Tensor::zeros([4, 10, 16], &device);
//! let (output, final_state) = cfc.forward(input, None, None);
//!
//! // output: [4, 10, 8] - sequence of outputs
//! // final_state: [4, 32] - final hidden state
//! ```
//!
//! ## Tensor Shapes
//!
//! ### Input Tensor (3D)
//!
//! | Format | Shape | Default |
//! |--------|-------|---------|
//! | Batch-first | `[batch, seq_len, features]` | ✓ Yes |
//! | Sequence-first | `[seq_len, batch, features]` | No |
//!
//! Use `.with_batch_first(false)` to switch to sequence-first format.
//!
//! ### Output Tensor
//!
//! | Setting | Shape | Description |
//! |---------|-------|-------------|
//! | `return_sequences=true` (default) | `[batch, seq_len, output_size]` | All timesteps |
//! | `return_sequences=false` | `[batch, 1, output_size]` | Last timestep only |
//!
//! ### Hidden State Tensor (2D)
//!
//! Shape: `[batch, hidden_size]`
//!
//! - `hidden_size` = `wiring.units()` (total neurons)
//! - Can be passed to preserve state across batches
//!
//! ## Common Patterns
//!
//! ### Sequence Classification (return last output only)
//!
//! ```ignore
//! let cfc = CfC::<Backend>::new(input_size, hidden_size, &device)
//!     .with_return_sequences(false);
//!
//! let (output, _) = cfc.forward(input, None, None);
//! // output: [batch, 1, hidden_size] - just the final output
//! ```
//!
//! ### Sequence-to-Sequence (return all outputs)
//!
//! ```ignore
//! let cfc = CfC::<Backend>::new(input_size, hidden_size, &device)
//!     .with_return_sequences(true);  // default
//!
//! let (output, _) = cfc.forward(input, None, None);
//! // output: [batch, seq_len, hidden_size] - output at every timestep
//! ```
//!
//! ### Stateful Processing (preserve hidden state)
//!
//! ```ignore
//! let cfc = CfC::<Backend>::new(input_size, hidden_size, &device);
//!
//! let (output1, state) = cfc.forward(batch1, None, None);
//! let (output2, state) = cfc.forward(batch2, Some(state), None);
//! let (output3, state) = cfc.forward(batch3, Some(state), None);
//! // State persists across batches
//! ```
//!
//! ### With NCP Wiring (sparse, interpretable)
//!
//! ```ignore
//! let mut wiring = AutoNCP::new(64, 10, 0.5, 42);
//! wiring.build(input_size);
//!
//! let cfc = CfC::<Backend>::with_wiring(input_size, wiring, &device);
//!
//! let (output, _) = cfc.forward(input, None, None);
//! // output: [batch, seq_len, 10] - projected to motor neurons
//! ```
//!
//! ## CfC vs LTC: When to Use Each
//!
//! ### Use CfC (Recommended) When:
//! - Speed is important
//! - Training large models
//! - Production deployment
//! - You don't need exact ODE solutions
//!
//! ### Use LTC When:
//! - Biological accuracy matters
//! - Research applications
//! - Comparing with neuroscience models
//! - You need variable time constants
//!
//! ## Mixed Memory (LSTM Augmentation)
//!
//! [`LTC`] supports "mixed memory" which augments the LTC cell with an LSTM
//! for improved long-term dependency handling:
//!
//! ```ignore
//! let ltc = LTC::<Backend>::new(input_size, wiring, &device)
//!     .with_mixed_memory(true, &device);
//!
//! // Use forward_mixed() instead of forward()
//! let (output, ltc_state, lstm_state) = ltc.forward_mixed(input, None, None, None);
//! ```

pub mod cfc;
pub mod ltc;

pub use cfc::CfC;
pub use ltc::LTC;
