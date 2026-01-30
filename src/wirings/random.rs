use super::base::Wiring;
use super::WiringConfig;
use ndarray::Array2;
use rand::prelude::*;

/// Random sparsity wiring structure
#[derive(Clone, Debug)]
pub struct Random {
    units: usize,
    output_dim: usize,
    adjacency_matrix: Array2<i32>,
    sensory_adjacency_matrix: Option<Array2<i32>>,
    input_dim: Option<usize>,
    sparsity_level: f64,
    random_seed: u64,
}

impl Random {
    pub fn new(
        units: usize,
        output_dim: Option<usize>,
        sparsity_level: f64,
        random_seed: u64,
    ) -> Self {
        if sparsity_level < 0.0 || sparsity_level >= 1.0 {
            panic!(
                "Sparsity level must be in range [0, 1), got {}",
                sparsity_level
            );
        }

        let output_dim = output_dim.unwrap_or(units);
        let mut adjacency_matrix = Array2::zeros((units, units));
        let mut rng = StdRng::seed_from_u64(random_seed);

        // Calculate number of synapses
        let total_possible = units * units;
        let num_synapses = (total_possible as f64 * (1.0 - sparsity_level)).round() as usize;

        // Create all possible synapse pairs
        let mut all_synapses: Vec<(usize, usize)> = Vec::with_capacity(total_possible);
        for src in 0..units {
            for dest in 0..units {
                all_synapses.push((src, dest));
            }
        }

        // Randomly select synapses
        let selected: Vec<_> = all_synapses
            .choose_multiple(&mut rng, num_synapses)
            .cloned()
            .collect();

        for (src, dest) in selected {
            let polarity: i32 = if rng.gen::<f64>() < 0.33 { -1 } else { 1 };
            adjacency_matrix[[src, dest]] = polarity;
        }

        Self {
            units,
            output_dim,
            adjacency_matrix,
            sensory_adjacency_matrix: None,
            input_dim: None,
            sparsity_level,
            random_seed,
        }
    }

    pub fn from_config(config: WiringConfig) -> Self {
        Self::new(
            config.units,
            config.output_dim,
            config.sparsity_level.unwrap_or(0.5),
            config.random_seed.unwrap_or(1111),
        )
    }
}

impl Wiring for Random {
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
        let mut sensory_matrix = Array2::zeros((input_dim, self.units));
        let mut rng = StdRng::seed_from_u64(self.random_seed);

        let total_possible = input_dim * self.units;
        let num_sensory_synapses =
            (total_possible as f64 * (1.0 - self.sparsity_level)).round() as usize;

        let mut all_sensory_synapses: Vec<(usize, usize)> = Vec::with_capacity(total_possible);
        for src in 0..input_dim {
            for dest in 0..self.units {
                all_sensory_synapses.push((src, dest));
            }
        }

        let selected: Vec<_> = all_sensory_synapses
            .choose_multiple(&mut rng, num_sensory_synapses)
            .cloned()
            .collect();

        for (src, dest) in selected {
            let polarity: i32 = if rng.gen::<f64>() < 0.33 { -1 } else { 1 };
            sensory_matrix[[src, dest]] = polarity;
        }

        self.sensory_adjacency_matrix = Some(sensory_matrix);
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
            output_dim: Some(self.output_dim),
            // Random-specific fields
            sparsity_level: Some(self.sparsity_level),
            random_seed: Some(self.random_seed),
            // Other fields not used by Random
            erev_init_seed: None,
            self_connections: None,
            num_inter_neurons: None,
            num_command_neurons: None,
            num_motor_neurons: None,
            sensory_fanout: None,
            inter_fanout: None,
            recurrent_command_synapses: None,
            motor_fanin: None,
            seed: None,
        }
    }
}
