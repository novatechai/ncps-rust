use rand::prelude::*;
use super::WiringConfig;

/// Base trait for wiring configurations in Neural Circuit Policies
pub trait Wiring: Send + Sync {
    /// Returns the number of neurons in this wiring
    fn units(&self) -> usize;

    /// Returns the input dimension (number of sensory neurons)
    fn input_dim(&self) -> Option<usize>;

    /// Returns the output dimension (number of motor neurons)
    fn output_dim(&self) -> Option<usize>;

    /// Returns the number of layers in this wiring
    fn num_layers(&self) -> usize {
        1
    }

    /// Returns the neuron IDs for a specific layer
    fn get_neurons_of_layer(&self, layer_id: usize) -> Vec<usize> {
        if layer_id == 0 {
            (0..self.units()).collect()
        } else {
            vec![]
        }
    }

    /// Check if the wiring has been built (input dimension is set)
    fn is_built(&self) -> bool {
        self.input_dim().is_some()
    }

    /// Build the wiring with the given input dimension
    fn build(&mut self, input_dim: usize);

    /// Get type of a neuron (motor, inter, command, etc.)
    fn get_type_of_neuron(&self, neuron_id: usize) -> &'static str {
        let output_dim = self.output_dim().unwrap_or(0);
        if neuron_id < output_dim {
            "motor"
        } else {
            "inter"
        }
    }

    /// Returns the adjacency matrix representing synapses between neurons
    fn adjacency_matrix(&self) -> &ndarray::Array2<i32>;

    /// Returns the sensory adjacency matrix (synapses from inputs to neurons)
    fn sensory_adjacency_matrix(&self) -> Option<&ndarray::Array2<i32>>;

    /// Initialize the adjacency matrix (erev)
    fn erev_initializer(&self) -> ndarray::Array2<i32> {
        self.adjacency_matrix().clone()
    }

    /// Initialize the sensory adjacency matrix (erev)
    fn sensory_erev_initializer(&self) -> Option<ndarray::Array2<i32>> {
        self.sensory_adjacency_matrix().map(|m| m.clone())
    }

    /// Add a synapse between neurons
    fn add_synapse(&mut self, src: usize, dest: usize, polarity: i32);

    /// Add a sensory synapse from input feature to neuron
    fn add_sensory_synapse(&mut self, src: usize, dest: usize, polarity: i32);

    /// Count total internal synapses
    fn synapse_count(&self) -> usize {
        self.adjacency_matrix().mapv(|x| x.abs() as usize).sum()
    }

    /// Count total sensory synapses
    fn sensory_synapse_count(&self) -> usize {
        self.sensory_adjacency_matrix()
            .map(|m| m.mapv(|x| x.abs() as usize).sum())
            .unwrap_or(0)
    }

    fn input_required(&self) -> bool {
        self.sensory_adjacency_matrix().is_some()
    }

    /// Create a serialization config for this wiring
    fn get_config(&self) -> WiringConfig;
}

/// Fully connected wiring structure
#[derive(Clone, Debug)]
pub struct FullyConnected {
    units: usize,
    output_dim: usize,
    adjacency_matrix: ndarray::Array2<i32>,
    sensory_adjacency_matrix: Option<ndarray::Array2<i32>>,
    input_dim: Option<usize>,
    self_connections: bool,
    erev_init_seed: u64,
}

impl FullyConnected {
    pub fn new(
        units: usize,
        output_dim: Option<usize>,
        erev_init_seed: u64,
        self_connections: bool,
    ) -> Self {
        let output_dim = output_dim.unwrap_or(units);
        let mut adjacency_matrix = ndarray::Array2::zeros((units, units));
        let mut rng = StdRng::seed_from_u64(erev_init_seed);

        // Initialize synapses
        for src in 0..units {
            for dest in 0..units {
                if src == dest && !self_connections {
                    continue;
                }
                // 2/3 chance of excitatory, 1/3 inhibitory
                let polarity: i32 = if rand::random::<f64>() < 0.33 { -1 } else { 1 };
                adjacency_matrix[[src, dest]] = polarity;
            }
        }

        Self {
            units,
            output_dim,
            adjacency_matrix,
            sensory_adjacency_matrix: None,
            input_dim: None,
            self_connections,
            erev_init_seed,
        }
    }

    /// Get configuration for serialization
    pub fn get_full_config(&self) -> WiringConfig {
        WiringConfig {
            units: self.units,
            adjacency_matrix: Some(
                self.adjacency_matrix
                    .outer_iter()
                    .map(|v| v.to_vec())
                    .collect(),
            ),
            sensory_adjacency_matrix: self
                .sensory_adjacency_matrix
                .as_ref()
                .map(|m| m.outer_iter().map(|v| v.to_vec()).collect()),
            input_dim: self.input_dim,
            output_dim: Some(self.output_dim),
            // FullyConnected-specific fields
            erev_init_seed: Some(self.erev_init_seed),
            self_connections: Some(self.self_connections),
            // Other fields not used by FullyConnected
            num_inter_neurons: None,
            num_command_neurons: None,
            num_motor_neurons: None,
            sensory_fanout: None,
            inter_fanout: None,
            recurrent_command_synapses: None,
            motor_fanin: None,
            seed: None,
            sparsity_level: None,
            random_seed: None,
        }
    }

    pub fn from_config(config: WiringConfig) -> Self {
        let units = config.units;
        let adjacency_matrix = if let Some(matrix) = config.adjacency_matrix {
            ndarray::Array2::from_shape_vec((units, units), matrix.into_iter().flatten().collect())
                .expect("Invalid adjacency matrix shape")
        } else {
            ndarray::Array2::zeros((units, units))
        };

        let sensory_adjacency_matrix = config.sensory_adjacency_matrix.map(|matrix| {
            let input_dim = config
                .input_dim
                .expect("Input dimension required when sensory matrix exists");
            ndarray::Array2::from_shape_vec(
                (input_dim, units),
                matrix.into_iter().flatten().collect(),
            )
            .expect("Invalid sensory adjacency matrix shape")
        });

        Self {
            units,
            output_dim: config.output_dim.unwrap_or(units),
            adjacency_matrix,
            sensory_adjacency_matrix,
            input_dim: config.input_dim,
            self_connections: true,
            erev_init_seed: 1111,
        }
    }
}

impl Wiring for FullyConnected {
    fn units(&self) -> usize {
        self.units
    }

    fn input_dim(&self) -> Option<usize> {
        self.input_dim
    }

    fn output_dim(&self) -> Option<usize> {
        Some(self.output_dim)
    }

    fn build(&mut self, input_dim: usize) {
        if let Some(existing) = self.input_dim {
            if existing != input_dim {
                panic!(
                    "Conflicting input dimensions: expected {}, got {}",
                    existing, input_dim
                );
            }
            return;
        }

        self.input_dim = Some(input_dim);
        let mut sensory_matrix = ndarray::Array2::zeros((input_dim, self.units));
        let mut rng = StdRng::seed_from_u64(self.erev_init_seed);

        for src in 0..input_dim {
            for dest in 0..self.units {
                let polarity: i32 = if rng.gen::<f64>() < 0.33 { -1 } else { 1 };
                sensory_matrix[[src, dest]] = polarity;
            }
        }
        self.sensory_adjacency_matrix = Some(sensory_matrix);
    }

    fn adjacency_matrix(&self) -> &ndarray::Array2<i32> {
        &self.adjacency_matrix
    }

    fn sensory_adjacency_matrix(&self) -> Option<&ndarray::Array2<i32>> {
        self.sensory_adjacency_matrix.as_ref()
    }

    fn add_synapse(&mut self, src: usize, dest: usize, polarity: i32) {
        if src >= self.units || dest >= self.units {
            panic!(
                "Invalid synapse: src={}, dest={}, units={}",
                src, dest, self.units
            );
        }
        if ![-1, 1].contains(&polarity) {
            panic!("Polarity must be -1 or 1, got {}", polarity);
        }
        self.adjacency_matrix[[src, dest]] = polarity;
    }

    fn add_sensory_synapse(&mut self, src: usize, dest: usize, polarity: i32) {
        let input_dim = self
            .input_dim
            .expect("Must build wiring before adding sensory synapses");
        if src >= input_dim || dest >= self.units {
            panic!(
                "Invalid sensory synapse: src={}, dest={}, input_dim={}, units={}",
                src, dest, input_dim, self.units
            );
        }
        if ![-1, 1].contains(&polarity) {
            panic!("Polarity must be -1 or 1, got {}", polarity);
        }
        self.sensory_adjacency_matrix.as_mut().unwrap()[[src, dest]] = polarity;
    }

    fn get_config(&self) -> WiringConfig {
        WiringConfig {
            units: self.units,
            adjacency_matrix: Some(
                self.adjacency_matrix
                    .outer_iter()
                    .map(|v| v.to_vec())
                    .collect(),
            ),
            sensory_adjacency_matrix: self
                .sensory_adjacency_matrix
                .as_ref()
                .map(|m| m.outer_iter().map(|v| v.to_vec()).collect()),
            input_dim: self.input_dim,
            output_dim: Some(self.output_dim),
            // FullyConnected-specific fields
            erev_init_seed: Some(self.erev_init_seed),
            self_connections: Some(self.self_connections),
            // Other fields not used by FullyConnected
            num_inter_neurons: None,
            num_command_neurons: None,
            num_motor_neurons: None,
            sensory_fanout: None,
            inter_fanout: None,
            recurrent_command_synapses: None,
            motor_fanin: None,
            seed: None,
            sparsity_level: None,
            random_seed: None,
        }
    }
}
