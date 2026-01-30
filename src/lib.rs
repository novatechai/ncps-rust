//! # NCPS - Neural Circuit Policies (Rust)
//!
//! Port of the Python NCPS library to Rust using the Burn framework.
//!
//! ## Features
//!
//! - **LTC**: Liquid Time-Constant RNN cells with ODE-based dynamics
//! - **CfC**: Closed-form Continuous-time cells (3 modes: default, pure, no_gate)
//! - **NCP**: Neural Circuit Policy wiring (biologically-inspired sparse connectivity)
//! - **Sparsity Masks**: Wiring adjacency properly enforced in forward passes
//! - **Input/Output Mapping**: Affine, Linear, or pass-through modes
//! - **Mixed Memory**: LSTM augmentation for long-term dependencies
//!
//! ## Quick Start
//!
//! ```rust
//! use ncps::prelude::*;
//!
//! // Create a wiring configuration
//! let mut wiring = AutoNCP::new(32, 8, 0.5, 22222);
//! wiring.build(16); // 16 input features
//!
//! assert_eq!(wiring.units(), 32);
//! assert_eq!(wiring.output_dim(), Some(8));
//! ```
//!
//! ## Cell-level Usage
//!
//! For direct cell access (single timestep processing):
//!
//! ```ignore
//! use ncps::cells::{LTCCell, MappingMode};
//! use ncps::wirings::FullyConnected;
//!
//! let wiring = FullyConnected::new(32, Some(8), 1234, true);
//! let cell = LTCCell::<Backend>::new(&wiring, Some(16), &device)
//!     .with_input_mapping(MappingMode::Affine, &device);
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
