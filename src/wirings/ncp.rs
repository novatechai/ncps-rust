use super::base::Wiring;
use super::WiringConfig;
use ndarray::Array2;
use rand::prelude::*;

/// Neural Circuit Policy wiring structure
/// Implements a 4-layer architecture: sensories -> inter -> command -> motor
#[derive(Clone, Debug)]
pub struct NCP {
    units: usize,
    adjacency_matrix: Array2<i32>,
    sensory_adjacency_matrix: Option<Array2<i32>>,
    input_dim: Option<usize>,
    num_inter_neurons: usize,
    num_command_neurons: usize,
    num_motor_neurons: usize,
    sensory_fanout: usize,
    inter_fanout: usize,
    recurrent_command_synapses: usize,
    motor_fanin: usize,
    motor_neurons: Vec<usize>,
    command_neurons: Vec<usize>,
    inter_neurons: Vec<usize>,
    sensory_neurons: Vec<usize>,
    rng: StdRng,
}

impl NCP {
    pub fn new(
        inter_neurons: usize,
        command_neurons: usize,
        motor_neurons: usize,
        sensory_fanout: usize,
        inter_fanout: usize,
        recurrent_command_synapses: usize,
        motor_fanin: usize,
        seed: u64,
    ) -> Self {
        let units = inter_neurons + command_neurons + motor_neurons;

        // Validate parameters
        if motor_fanin > command_neurons {
            panic!(
                "Motor fanin {} exceeds number of command neurons {}",
                motor_fanin, command_neurons
            );
        }
        if sensory_fanout > inter_neurons {
            panic!(
                "Sensory fanout {} exceeds number of inter neurons {}",
                sensory_fanout, inter_neurons
            );
        }
        if inter_fanout > command_neurons {
            panic!(
                "Inter fanout {} exceeds number of command neurons {}",
                inter_fanout, command_neurons
            );
        }

        // Neuron IDs: [0..motor ... command ... inter]
        let motor_neuron_ids: Vec<usize> = (0..motor_neurons).collect();
        let command_neuron_ids: Vec<usize> =
            (motor_neurons..motor_neurons + command_neurons).collect();
        let inter_neuron_ids: Vec<usize> = (motor_neurons + command_neurons..units).collect();

        let adjacency_matrix = Array2::zeros((units, units));
        let rng = StdRng::seed_from_u64(seed);

        Self {
            units,
            adjacency_matrix,
            sensory_adjacency_matrix: None,
            input_dim: None,
            num_inter_neurons: inter_neurons,
            num_command_neurons: command_neurons,
            num_motor_neurons: motor_neurons,
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin,
            motor_neurons: motor_neuron_ids,
            command_neurons: command_neuron_ids,
            inter_neurons: inter_neuron_ids,
            sensory_neurons: vec![],
            rng,
        }
    }

    fn build_sensory_to_inter_layer(&mut self) {
        let input_dim = self.input_dim.unwrap();
        self.sensory_neurons = (0..input_dim).collect();

        // Clone the neuron list to avoid borrow issues
        let inter_neurons = self.inter_neurons.clone();
        let sensory_neurons = self.sensory_neurons.clone();
        let mut unreachable_inter: Vec<usize> = inter_neurons.clone();

        // Connect each sensory neuron to exactly sensory_fanout inter neurons
        for &src in &sensory_neurons {
            let selected: Vec<_> = inter_neurons
                .choose_multiple(&mut self.rng, self.sensory_fanout)
                .cloned()
                .collect();

            for &dest in &selected {
                if let Some(pos) = unreachable_inter.iter().position(|&x| x == dest) {
                    unreachable_inter.remove(pos);
                }
                let polarity: i32 = if self.rng.gen::<bool>() { 1 } else { -1 };
                self.add_sensory_synapse(src, dest, polarity);
            }
        }

        // Connect any unreachable inter neurons
        let mean_inter_fanin = (input_dim * self.sensory_fanout / self.num_inter_neurons)
            .max(1)
            .min(input_dim);

        for &dest in &unreachable_inter {
            let selected: Vec<_> = sensory_neurons
                .choose_multiple(&mut self.rng, mean_inter_fanin)
                .cloned()
                .collect();

            for &src in &selected {
                let polarity: i32 = if self.rng.gen::<bool>() { 1 } else { -1 };
                self.add_sensory_synapse(src, dest, polarity);
            }
        }
    }

    fn build_inter_to_command_layer(&mut self) {
        // Clone the neuron lists to avoid borrow issues
        let inter_neurons = self.inter_neurons.clone();
        let command_neurons = self.command_neurons.clone();
        let mut unreachable_command: Vec<usize> = command_neurons.clone();

        // Connect inter neurons to command neurons
        for &src in &inter_neurons {
            let selected: Vec<_> = command_neurons
                .choose_multiple(&mut self.rng, self.inter_fanout)
                .cloned()
                .collect();

            for &dest in &selected {
                if let Some(pos) = unreachable_command.iter().position(|&x| x == dest) {
                    unreachable_command.remove(pos);
                }
                let polarity: i32 = if self.rng.gen::<bool>() { 1 } else { -1 };
                self.add_synapse(src, dest, polarity);
            }
        }

        // Connect any unreachable command neurons
        let mean_command_fanin = (self.num_inter_neurons * self.inter_fanout
            / self.num_command_neurons)
            .max(1)
            .min(self.num_inter_neurons);

        for &dest in &unreachable_command {
            let selected: Vec<_> = inter_neurons
                .choose_multiple(&mut self.rng, mean_command_fanin)
                .cloned()
                .collect();

            for &src in &selected {
                let polarity: i32 = if self.rng.gen::<bool>() { 1 } else { -1 };
                self.add_synapse(src, dest, polarity);
            }
        }
    }

    fn build_recurrent_command_layer(&mut self) {
        for _ in 0..self.recurrent_command_synapses {
            let src = *self.command_neurons.choose(&mut self.rng).unwrap();
            let dest = *self.command_neurons.choose(&mut self.rng).unwrap();
            let polarity: i32 = if self.rng.gen::<bool>() { 1 } else { -1 };
            self.add_synapse(src, dest, polarity);
        }
    }

    fn build_command_to_motor_layer(&mut self) {
        // Clone the neuron lists to avoid borrow issues
        let motor_neurons = self.motor_neurons.clone();
        let command_neurons = self.command_neurons.clone();
        let mut unreachable_command: Vec<usize> = command_neurons.clone();

        // Connect command neurons to motor neurons
        for &dest in &motor_neurons {
            let selected: Vec<_> = command_neurons
                .choose_multiple(&mut self.rng, self.motor_fanin)
                .cloned()
                .collect();

            for &src in &selected {
                if let Some(pos) = unreachable_command.iter().position(|&x| x == src) {
                    unreachable_command.remove(pos);
                }
                let polarity: i32 = if self.rng.gen::<bool>() { 1 } else { -1 };
                self.add_synapse(src, dest, polarity);
            }
        }

        // Connect any unreachable command neurons
        let mean_command_fanout = (self.num_motor_neurons * self.motor_fanin
            / self.num_command_neurons)
            .max(1)
            .min(self.num_motor_neurons);

        for &src in &unreachable_command {
            let selected: Vec<_> = motor_neurons
                .choose_multiple(&mut self.rng, mean_command_fanout)
                .cloned()
                .collect();

            for &dest in &selected {
                let polarity: i32 = if self.rng.gen::<bool>() { 1 } else { -1 };
                self.add_synapse(src, dest, polarity);
            }
        }
    }

    pub fn from_config(config: WiringConfig) -> Self {
        // Parse config to reconstruct NCP
        let units = config.units;
        let adjacency_matrix = if let Some(matrix) = config.adjacency_matrix {
            Array2::from_shape_vec((units, units), matrix.into_iter().flatten().collect())
                .expect("Invalid adjacency matrix shape")
        } else {
            Array2::zeros((units, units))
        };

        let sensory_adjacency_matrix = config.sensory_adjacency_matrix.map(|matrix| {
            let input_dim = config
                .input_dim
                .expect("Input dimension required when sensory matrix exists");
            Array2::from_shape_vec((input_dim, units), matrix.into_iter().flatten().collect())
                .expect("Invalid sensory adjacency matrix shape")
        });

        // This would need additional info stored in config to reconstruct properly
        // For now, create a basic NCP structure
        let output_dim = config.output_dim.unwrap_or(1);
        let inter_and_command = units - output_dim;
        let command_neurons = (inter_and_command as f64 * 0.4).ceil() as usize;
        let inter_neurons = inter_and_command - command_neurons;

        NCP::new(
            inter_neurons,
            command_neurons,
            output_dim,
            6,     // Default sensory_fanout
            6,     // Default inter_fanout
            4,     // Default recurrent_command_synapses
            6,     // Default motor_fanin
            22222, // Default seed
        )
    }
}

impl Wiring for NCP {
    fn units(&self) -> usize {
        self.units
    }

    fn input_dim(&self) -> Option<usize> {
        self.input_dim
    }

    fn output_dim(&self) -> Option<usize> {
        Some(self.num_motor_neurons)
    }

    fn num_layers(&self) -> usize {
        3
    }

    fn get_neurons_of_layer(&self, layer_id: usize) -> Vec<usize> {
        match layer_id {
            0 => self.inter_neurons.clone(),
            1 => self.command_neurons.clone(),
            2 => self.motor_neurons.clone(),
            _ => panic!("Unknown layer {}", layer_id),
        }
    }

    fn get_type_of_neuron(&self, neuron_id: usize) -> &'static str {
        if neuron_id < self.num_motor_neurons {
            "motor"
        } else if neuron_id < self.num_motor_neurons + self.num_command_neurons {
            "command"
        } else {
            "inter"
        }
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
        self.sensory_adjacency_matrix = Some(Array2::zeros((input_dim, self.units)));

        self.build_sensory_to_inter_layer();
        self.build_inter_to_command_layer();
        self.build_recurrent_command_layer();
        self.build_command_to_motor_layer();
    }

    fn adjacency_matrix(&self) -> &Array2<i32> {
        &self.adjacency_matrix
    }

    fn sensory_adjacency_matrix(&self) -> Option<&Array2<i32>> {
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
            output_dim: Some(self.num_motor_neurons),
            // NCP-specific fields
            num_inter_neurons: Some(self.num_inter_neurons),
            num_command_neurons: Some(self.num_command_neurons),
            num_motor_neurons: Some(self.num_motor_neurons),
            sensory_fanout: Some(self.sensory_fanout),
            inter_fanout: Some(self.inter_fanout),
            recurrent_command_synapses: Some(self.recurrent_command_synapses),
            motor_fanin: Some(self.motor_fanin),
            seed: None, // NCP uses rng internally
            // Other fields not used by NCP
            erev_init_seed: None,
            self_connections: None,
            sparsity_level: None,
            random_seed: None,
        }
    }
}

/// AutoNCP provides an easier way to create NCP wiring
#[derive(Clone, Debug)]
pub struct AutoNCP {
    ncp: NCP,
    output_size: usize,
    sparsity_level: f64,
    seed: u64,
}

impl AutoNCP {
    pub fn new(units: usize, output_size: usize, sparsity_level: f64, seed: u64) -> Self {
        if output_size >= units - 2 {
            panic!(
                "Output size {} must be less than units-2 ({})",
                output_size,
                units - 2
            );
        }
        if sparsity_level < 0.0 || sparsity_level > 0.9 {
            panic!(
                "Sparsity level must be between 0.0 and 0.9, got {}",
                sparsity_level
            );
        }

        let density_level = 1.0 - sparsity_level;
        let inter_and_command_neurons = units - output_size;
        let command_neurons = ((inter_and_command_neurons as f64 * 0.4).ceil() as usize).max(1);
        let inter_neurons = inter_and_command_neurons - command_neurons;

        let sensory_fanout = ((inter_neurons as f64 * density_level).ceil() as usize).max(1);
        let inter_fanout = ((command_neurons as f64 * density_level).ceil() as usize).max(1);
        let recurrent_command_synapses =
            ((command_neurons as f64 * density_level * 2.0).ceil() as usize).max(1);
        let motor_fanin = ((command_neurons as f64 * density_level).ceil() as usize).max(1);

        let ncp = NCP::new(
            inter_neurons,
            command_neurons,
            output_size,
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin,
            seed,
        );

        Self {
            ncp,
            output_size,
            sparsity_level,
            seed,
        }
    }
}

impl Wiring for AutoNCP {
    fn units(&self) -> usize {
        self.ncp.units()
    }

    fn input_dim(&self) -> Option<usize> {
        self.ncp.input_dim()
    }

    fn output_dim(&self) -> Option<usize> {
        Some(self.output_size)
    }

    fn num_layers(&self) -> usize {
        self.ncp.num_layers()
    }

    fn get_neurons_of_layer(&self, layer_id: usize) -> Vec<usize> {
        self.ncp.get_neurons_of_layer(layer_id)
    }

    fn get_type_of_neuron(&self, neuron_id: usize) -> &'static str {
        self.ncp.get_type_of_neuron(neuron_id)
    }

    fn build(&mut self, input_dim: usize) {
        self.ncp.build(input_dim)
    }

    fn is_built(&self) -> bool {
        self.ncp.is_built()
    }

    fn adjacency_matrix(&self) -> &Array2<i32> {
        self.ncp.adjacency_matrix()
    }

    fn sensory_adjacency_matrix(&self) -> Option<&Array2<i32>> {
        self.ncp.sensory_adjacency_matrix()
    }

    fn add_synapse(&mut self, src: usize, dest: usize, polarity: i32) {
        self.ncp.add_synapse(src, dest, polarity)
    }

    fn add_sensory_synapse(&mut self, src: usize, dest: usize, polarity: i32) {
        self.ncp.add_sensory_synapse(src, dest, polarity)
    }

    fn get_config(&self) -> WiringConfig {
        // Get the underlying NCP config and add AutoNCP-specific fields
        let mut config = self.ncp.get_config();
        config.output_dim = Some(self.output_size);
        config.sparsity_level = Some(self.sparsity_level);
        config.seed = Some(self.seed);
        config
    }
}
