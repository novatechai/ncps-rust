//! # Wiring Configurations for Neural Circuit Policies
//!
//! This module provides different connectivity patterns (wirings) that define how neurons
//! connect to each other in Neural Circuit Policy networks. Wiring determines the **sparsity**
//! and **structure** of the neural network.
//!
//! ## Key Concepts
//!
//! ### What is Wiring?
//!
//! Traditional RNNs use fully-connected layers where every neuron connects to every other neuron.
//! NCPs use **sparse, structured connectivity** inspired by biological neural circuits (specifically
//! the nervous system of *C. elegans*, a nematode with only 302 neurons).
//!
//! Wiring defines:
//! - **Which neurons connect** to which other neurons (adjacency matrix)
//! - **Synapse polarity** (+1 excitatory, -1 inhibitory)
//! - **Layer structure** (sensory → inter → command → motor in NCP)
//!
//! ### The `.build()` Method - IMPORTANT!
//!
//! **You must call `.build(input_dim)` before using any wiring with an RNN layer.**
//!
//! ```rust
//! use ncps::wirings::{AutoNCP, Wiring};
//!
//! // Create wiring configuration
//! let mut wiring = AutoNCP::new(32, 8, 0.5, 42);
//!
//! // REQUIRED: Build with input dimension before use
//! wiring.build(16);  // 16 input features
//!
//! // Now the wiring is ready
//! assert!(wiring.is_built());
//! assert_eq!(wiring.input_dim(), Some(16));
//! ```
//!
//! **Why is `.build()` needed?**
//! - The sensory adjacency matrix (input → neurons) can only be created once we know the input size
//! - This allows the same wiring type to work with different input dimensions
//! - Calling `.build()` a second time with the same dimension is a no-op; with a different dimension it panics
//!
//! ## Choosing a Wiring Type
//!
//! | Wiring | Use Case | Sparsity | Structure |
//! |--------|----------|----------|-----------|
//! | [`AutoNCP`] | **Recommended default** - automatic parameter selection | Medium-High | 4-layer biological |
//! | [`NCP`] | Fine-grained control over layer sizes and connectivity | Configurable | 4-layer biological |
//! | [`FullyConnected`] | Baseline comparison, maximum expressiveness | None (dense) | Single layer |
//! | [`Random`] | Unstructured sparse networks for ablation studies | Configurable | Random |
//!
//! ## NCP Architecture (4-Layer Biological Structure)
//!
//! ```text
//!                    ┌─────────────────────────────────────────┐
//!                    │           Motor Neurons (output)        │
//!                    │  - Final output layer                   │
//!                    │  - Size = output_dim                    │
//!                    └────────────────▲────────────────────────┘
//!                                     │ motor_fanin connections
//!                    ┌────────────────┴────────────────────────┐
//!                    │          Command Neurons                │
//!                    │  - Decision/integration layer           │
//!                    │  - Has recurrent connections            │
//!                    └────────────────▲────────────────────────┘
//!                                     │ inter_fanout connections
//!                    ┌────────────────┴────────────────────────┐
//!                    │           Inter Neurons                 │
//!                    │  - Feature processing layer             │
//!                    │  - Receives from sensory inputs         │
//!                    └────────────────▲────────────────────────┘
//!                                     │ sensory_fanout connections
//!                    ┌────────────────┴────────────────────────┐
//!                    │         Sensory Inputs (input)          │
//!                    │  - External input features              │
//!                    │  - Size = input_dim (set by .build())   │
//!                    └─────────────────────────────────────────┘
//! ```
//!
//! ## Quick Examples
//!
//! ### Using AutoNCP (Recommended)
//!
//! ```rust
//! use ncps::wirings::{AutoNCP, Wiring};
//!
//! // 32 total neurons, 8 motor outputs, 50% sparsity
//! let mut wiring = AutoNCP::new(32, 8, 0.5, 42);
//! wiring.build(16);  // 16 input features
//!
//! println!("Total neurons: {}", wiring.units());        // 32
//! println!("Output size: {:?}", wiring.output_dim());   // Some(8)
//! println!("Internal synapses: {}", wiring.synapse_count());
//! println!("Sensory synapses: {}", wiring.sensory_synapse_count());
//! ```
//!
//! ### Using Manual NCP Configuration
//!
//! ```rust
//! use ncps::wirings::{NCP, Wiring};
//!
//! // Fine-grained control: 10 inter, 8 command, 4 motor neurons
//! let mut wiring = NCP::new(
//!     10,  // inter_neurons
//!     8,   // command_neurons
//!     4,   // motor_neurons (output)
//!     4,   // sensory_fanout: each sensory connects to 4 inter neurons
//!     4,   // inter_fanout: each inter connects to 4 command neurons
//!     6,   // recurrent_command_synapses
//!     4,   // motor_fanin: each motor receives from 4 command neurons
//!     42,  // seed for reproducibility
//! );
//! wiring.build(16);
//! ```
//!
//! ### Inspecting Connectivity
//!
//! ```rust
//! use ncps::wirings::{AutoNCP, Wiring};
//!
//! let mut wiring = AutoNCP::new(32, 8, 0.5, 42);
//! wiring.build(16);
//!
//! // Check neuron types
//! for i in 0..wiring.units() {
//!     let neuron_type = wiring.get_type_of_neuron(i);
//!     // Returns "motor", "command", or "inter"
//! }
//!
//! // Access adjacency matrices for visualization
//! let adj = wiring.adjacency_matrix();  // [units x units]
//! let sensory_adj = wiring.sensory_adjacency_matrix();  // [input_dim x units]
//! ```

use serde::{Deserialize, Serialize};

mod base;
mod ncp;
mod random;

pub use base::{FullyConnected, Wiring, Wiring as WiringTrait};
pub use ncp::{AutoNCP, NCP};
pub use random::Random;

/// Configuration struct for serialization/deserialization of wiring structures
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WiringConfig {
    pub units: usize,
    pub adjacency_matrix: Option<Vec<Vec<i32>>>,
    pub sensory_adjacency_matrix: Option<Vec<Vec<i32>>>,
    pub input_dim: Option<usize>,
    pub output_dim: Option<usize>,
    // FullyConnected fields
    pub erev_init_seed: Option<u64>,
    pub self_connections: Option<bool>,
    // NCP fields
    pub num_inter_neurons: Option<usize>,
    pub num_command_neurons: Option<usize>,
    pub num_motor_neurons: Option<usize>,
    pub sensory_fanout: Option<usize>,
    pub inter_fanout: Option<usize>,
    pub recurrent_command_synapses: Option<usize>,
    pub motor_fanin: Option<usize>,
    pub seed: Option<u64>,
    // Random fields
    pub sparsity_level: Option<f64>,
    pub random_seed: Option<u64>,
}
