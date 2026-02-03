use rand::prelude::*;
use super::WiringConfig;

/// Core trait defining connectivity patterns for Neural Circuit Policy networks.
///
/// The `Wiring` trait specifies how neurons connect to each other and to external inputs.
/// It defines the sparse connectivity structure that makes NCPs parameter-efficient and
/// interpretable compared to fully-connected RNNs.
///
/// # Lifecycle
///
/// 1. **Create** the wiring with desired structure (e.g., `AutoNCP::new(...)`)
/// 2. **Build** with input dimension: `wiring.build(input_dim)` - this creates the sensory connections
/// 3. **Use** with an RNN layer: `CfC::with_wiring(input_size, wiring, device)`
///
/// # Key Matrices
///
/// - **Adjacency Matrix** `[units × units]`: Internal neuron-to-neuron connections
///   - Values: +1 (excitatory), -1 (inhibitory), 0 (no connection)
/// - **Sensory Adjacency Matrix** `[input_dim × units]`: Input-to-neuron connections
///   - Only available after calling `.build()`
///
/// # Example
///
/// ```rust
/// use ncps::wirings::{AutoNCP, Wiring};
///
/// let mut wiring = AutoNCP::new(32, 8, 0.5, 42);
///
/// // Check state before building
/// assert!(!wiring.is_built());
/// assert_eq!(wiring.input_dim(), None);
///
/// // Build with input dimension
/// wiring.build(16);
///
/// // Now fully configured
/// assert!(wiring.is_built());
/// assert_eq!(wiring.input_dim(), Some(16));
/// assert_eq!(wiring.units(), 32);
/// assert_eq!(wiring.output_dim(), Some(8));
/// ```
///
/// # Implementors
///
/// - [`super::AutoNCP`]: Automatic NCP configuration (recommended)
/// - [`super::NCP`]: Manual NCP configuration with full control
/// - [`FullyConnected`]: Dense connectivity (no sparsity)
/// - [`super::Random`]: Random sparse connectivity
pub trait Wiring: Send + Sync {
    /// Returns the total number of neurons (hidden units) in this wiring.
    ///
    /// This is the size of the hidden state tensor: `[batch, units]`.
    /// For NCP wirings, this equals `inter_neurons + command_neurons + motor_neurons`.
    fn units(&self) -> usize;

    /// Returns the input dimension (number of input features), or `None` if not yet built.
    ///
    /// This is only available after calling [`.build(input_dim)`](Wiring::build).
    /// The sensory adjacency matrix will have shape `[input_dim, units]`.
    fn input_dim(&self) -> Option<usize>;

    /// Returns the output dimension (number of motor neurons).
    ///
    /// For NCP wirings, this is the number of motor neurons. The RNN output
    /// will be projected to this size if it differs from `units()`.
    ///
    /// Returns `None` if output_dim equals units (no projection needed).
    fn output_dim(&self) -> Option<usize>;

    /// Returns the number of logical layers in this wiring.
    ///
    /// - `FullyConnected`: 1 layer
    /// - `NCP`/`AutoNCP`: 3 layers (inter, command, motor)
    fn num_layers(&self) -> usize {
        1
    }

    /// Returns the neuron IDs belonging to a specific layer.
    ///
    /// For NCP wirings:
    /// - Layer 0: Inter neurons (feature processing)
    /// - Layer 1: Command neurons (integration/decision)
    /// - Layer 2: Motor neurons (output)
    ///
    /// # Panics
    ///
    /// Panics if `layer_id >= num_layers()`.
    fn get_neurons_of_layer(&self, layer_id: usize) -> Vec<usize> {
        if layer_id == 0 {
            (0..self.units()).collect()
        } else {
            vec![]
        }
    }

    /// Returns `true` if the wiring has been built (input dimension is set).
    ///
    /// A wiring must be built before it can be used with an RNN layer.
    /// Call [`.build(input_dim)`](Wiring::build) to build the wiring.
    fn is_built(&self) -> bool {
        self.input_dim().is_some()
    }

    /// Builds the wiring by setting the input dimension and creating sensory connections.
    ///
    /// **This method must be called before using the wiring with an RNN layer.**
    ///
    /// The build process:
    /// 1. Sets the input dimension
    /// 2. Creates the sensory adjacency matrix `[input_dim × units]`
    /// 3. Establishes connections from inputs to the first layer of neurons
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Number of input features per timestep
    ///
    /// # Panics
    ///
    /// Panics if called twice with different input dimensions:
    /// ```should_panic
    /// use ncps::wirings::{AutoNCP, Wiring};
    ///
    /// let mut wiring = AutoNCP::new(32, 8, 0.5, 42);
    /// wiring.build(16);
    /// wiring.build(32);  // Panics! Different input_dim
    /// ```
    ///
    /// Calling with the same dimension twice is safe (no-op).
    fn build(&mut self, input_dim: usize);

    /// Returns the type of a neuron by its ID.
    ///
    /// # Returns
    ///
    /// - `"motor"`: Output neurons (IDs 0..motor_neurons)
    /// - `"command"`: Integration neurons (NCP only)
    /// - `"inter"`: Feature processing neurons
    ///
    /// # Example
    ///
    /// ```rust
    /// use ncps::wirings::{AutoNCP, Wiring};
    ///
    /// let mut wiring = AutoNCP::new(32, 8, 0.5, 42);
    /// wiring.build(16);
    ///
    /// // First 8 neurons are motor neurons (output)
    /// assert_eq!(wiring.get_type_of_neuron(0), "motor");
    /// assert_eq!(wiring.get_type_of_neuron(7), "motor");
    ///
    /// // Higher IDs are command or inter neurons
    /// let neuron_type = wiring.get_type_of_neuron(15);
    /// assert!(neuron_type == "command" || neuron_type == "inter");
    /// ```
    fn get_type_of_neuron(&self, neuron_id: usize) -> &'static str {
        let output_dim = self.output_dim().unwrap_or(0);
        if neuron_id < output_dim {
            "motor"
        } else {
            "inter"
        }
    }

    /// Returns the internal adjacency matrix representing neuron-to-neuron synapses.
    ///
    /// Shape: `[units × units]`
    ///
    /// Values:
    /// - `+1`: Excitatory synapse (increases activation)
    /// - `-1`: Inhibitory synapse (decreases activation)
    /// - `0`: No connection
    ///
    /// The matrix is indexed as `[source, destination]`, meaning `matrix[[i, j]]`
    /// represents a synapse from neuron `i` to neuron `j`.
    fn adjacency_matrix(&self) -> &ndarray::Array2<i32>;

    /// Returns the sensory adjacency matrix (input-to-neuron connections).
    ///
    /// Shape: `[input_dim × units]`
    ///
    /// Returns `None` before [`.build()`](Wiring::build) is called.
    ///
    /// The matrix is indexed as `[input_feature, neuron]`, meaning `matrix[[i, j]]`
    /// represents a synapse from input feature `i` to neuron `j`.
    fn sensory_adjacency_matrix(&self) -> Option<&ndarray::Array2<i32>>;

    /// Returns the reversal potential initializer (same as adjacency matrix).
    ///
    /// Used internally by LTC cells for biologically-plausible dynamics.
    fn erev_initializer(&self) -> ndarray::Array2<i32> {
        self.adjacency_matrix().clone()
    }

    /// Returns the sensory reversal potential initializer.
    ///
    /// Used internally by LTC cells for biologically-plausible dynamics.
    fn sensory_erev_initializer(&self) -> Option<ndarray::Array2<i32>> {
        self.sensory_adjacency_matrix().map(|m| m.clone())
    }

    /// Adds or modifies an internal synapse between two neurons.
    ///
    /// # Arguments
    ///
    /// * `src` - Source neuron ID (0..units)
    /// * `dest` - Destination neuron ID (0..units)
    /// * `polarity` - Synapse type: +1 (excitatory) or -1 (inhibitory)
    ///
    /// # Panics
    ///
    /// - Panics if `src >= units` or `dest >= units`
    /// - Panics if `polarity` is not +1 or -1
    fn add_synapse(&mut self, src: usize, dest: usize, polarity: i32);

    /// Adds or modifies a sensory synapse from an input feature to a neuron.
    ///
    /// # Arguments
    ///
    /// * `src` - Input feature index (0..input_dim)
    /// * `dest` - Destination neuron ID (0..units)
    /// * `polarity` - Synapse type: +1 (excitatory) or -1 (inhibitory)
    ///
    /// # Panics
    ///
    /// - Panics if wiring is not built (call `.build()` first)
    /// - Panics if `src >= input_dim` or `dest >= units`
    /// - Panics if `polarity` is not +1 or -1
    fn add_sensory_synapse(&mut self, src: usize, dest: usize, polarity: i32);

    /// Returns the total number of internal synapses (non-zero entries in adjacency matrix).
    ///
    /// This is a measure of model complexity/sparsity. Lower values indicate sparser networks.
    fn synapse_count(&self) -> usize {
        self.adjacency_matrix().mapv(|x| x.abs() as usize).sum()
    }

    /// Returns the total number of sensory synapses (input-to-neuron connections).
    ///
    /// Returns 0 if the wiring hasn't been built yet.
    fn sensory_synapse_count(&self) -> usize {
        self.sensory_adjacency_matrix()
            .map(|m| m.mapv(|x| x.abs() as usize).sum())
            .unwrap_or(0)
    }

    /// Returns `true` if this wiring requires external input (has sensory connections).
    ///
    /// Always returns `true` after `.build()` is called.
    fn input_required(&self) -> bool {
        self.sensory_adjacency_matrix().is_some()
    }

    /// Creates a serializable configuration for this wiring.
    ///
    /// Used for saving/loading models. See [`WiringConfig`] for details.
    fn get_config(&self) -> WiringConfig;
}

/// Fully connected (dense) wiring structure.
///
/// Every neuron connects to every other neuron (and optionally to itself).
/// This provides a baseline comparison for NCP's sparse connectivity.
///
/// # When to Use
///
/// - **Baseline comparison**: Compare NCP performance against dense networks
/// - **Maximum expressiveness**: When sparsity is not a concern
/// - **Debugging**: Simpler structure for testing
///
/// # Sparsity
///
/// `FullyConnected` has **no sparsity** - the adjacency matrix is fully populated.
/// For a network with `N` units, this means `N²` internal synapses (or `N²-N` without self-connections).
///
/// # Example
///
/// ```rust
/// use ncps::wirings::{FullyConnected, Wiring};
///
/// // Create a fully-connected wiring with 32 neurons, 8 outputs
/// let mut wiring = FullyConnected::new(
///     32,        // units (total neurons)
///     Some(8),   // output_dim (motor neurons)
///     42,        // seed for reproducibility
///     true,      // self_connections allowed
/// );
///
/// // Build with input dimension
/// wiring.build(16);
///
/// // Check connectivity
/// println!("Total synapses: {}", wiring.synapse_count());  // 32*32 = 1024
/// println!("Sensory synapses: {}", wiring.sensory_synapse_count());  // 16*32 = 512
/// ```
///
/// # Comparison with NCP
///
/// | Aspect | FullyConnected | NCP |
/// |--------|----------------|-----|
/// | Synapses | O(N²) | O(N) to O(N log N) |
/// | Interpretability | Low | High |
/// | Parameters | More | Fewer |
/// | Structure | Single layer | 4-layer biological |
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
