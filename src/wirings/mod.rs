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
