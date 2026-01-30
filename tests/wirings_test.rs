//! Tests for the wirings module

use ncps::wirings::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fully_connected_creation() {
        let fc = FullyConnected::new(10, None, 1234, true);
        assert_eq!(fc.units(), 10);
        assert_eq!(fc.output_dim(), Some(10));
        assert!(fc.is_built() == false);
    }

    #[test]
    fn test_fully_connected_build() {
        let mut fc = FullyConnected::new(10, Some(5), 1234, true);
        fc.build(20); // 20 input features
        assert!(fc.is_built());
        assert_eq!(fc.input_dim(), Some(20));

        // Check sensory matrix dimensions
        let sensory = fc.sensory_adjacency_matrix().unwrap();
        assert_eq!(sensory.shape(), &[20, 10]);

        // Check adjacency matrix has synapses
        let adj = fc.adjacency_matrix();
        let synapse_count = adj.iter().map(|&x| x.abs()).sum::<i32>();
        assert!(synapse_count > 0);
    }

    #[test]
    fn test_fully_connected_serialization() {
        let mut fc = FullyConnected::new(10, Some(5), 1234, true);
        fc.build(20);

        let config = fc.get_config();
        let fc2 = FullyConnected::from_config(config);

        assert_eq!(fc.units(), fc2.units());
        assert_eq!(fc.input_dim(), fc2.input_dim());
        assert_eq!(fc.output_dim(), fc2.output_dim());
    }

    #[test]
    #[should_panic]
    fn test_conflicting_input_dim() {
        let mut fc = FullyConnected::new(10, None, 1234, true);
        fc.build(20);
        fc.build(30); // Should panic
    }

    #[test]
    fn test_ncp_structure() {
        let ncp = NCP::new(
            10,    // inter_neurons
            8,     // command_neurons
            5,     // motor_neurons
            6,     // sensory_fanout
            6,     // inter_fanout
            4,     // recurrent_command_synapses
            6,     // motor_fanin
            22222, // seed
        );

        assert_eq!(ncp.units(), 23); // 10 + 8 + 5
        assert_eq!(ncp.output_dim(), Some(5));
        assert_eq!(ncp.num_layers(), 3);
        assert_eq!(ncp.get_neurons_of_layer(2).len(), 5); // motor
    }

    #[test]
    fn test_ncp_neuron_types() {
        let ncp = NCP::new(10, 8, 5, 6, 6, 4, 6, 22222);

        // First 5 neurons are motor
        assert_eq!(ncp.get_type_of_neuron(0), "motor");
        assert_eq!(ncp.get_type_of_neuron(4), "motor");

        // Next 8 neurons are command
        assert_eq!(ncp.get_type_of_neuron(5), "command");
        assert_eq!(ncp.get_type_of_neuron(12), "command");

        // Remaining are inter
        assert_eq!(ncp.get_type_of_neuron(13), "inter");
    }

    #[test]
    fn test_ncp_build() {
        let mut ncp = NCP::new(10, 8, 5, 6, 6, 4, 6, 22222);
        ncp.build(15); // 15 sensory inputs

        assert!(ncp.is_built());
        assert_eq!(ncp.input_dim(), Some(15));

        // Check connections were created
        let sensory = ncp.sensory_adjacency_matrix().unwrap();
        let sensory_synapses = sensory.iter().map(|&x| x.abs()).sum::<i32>();
        assert!(sensory_synapses > 0);
    }

    #[test]
    fn test_auto_ncp_convenience() {
        let auto_ncp = AutoNCP::new(32, 8, 0.5, 22222);

        assert_eq!(auto_ncp.units(), 32);
        assert_eq!(auto_ncp.output_dim(), Some(8));
        assert_eq!(auto_ncp.num_layers(), 3);
    }

    #[test]
    #[should_panic]
    fn test_auto_ncp_invalid_sparsity() {
        AutoNCP::new(32, 8, 1.5, 22222);
    }

    #[test]
    #[should_panic]
    fn test_auto_ncp_invalid_output_size() {
        AutoNCP::new(10, 9, 0.5, 22222); // 9 >= 10 - 2
    }

    #[test]
    fn test_wiring_synapse_count() {
        let mut fc = FullyConnected::new(10, None, 1234, true);
        fc.build(5);

        let count = fc.synapse_count();
        let adj = fc.adjacency_matrix();
        let manual_count: usize = adj.iter().map(|&x| x.abs() as usize).sum();
        assert_eq!(count, manual_count);
    }

    #[test]
    fn test_random_wiring() {
        let mut random = Random::new(10, Some(5), 0.5, 1234);
        random.build(20);

        assert_eq!(random.units(), 10);
        // With 0.5 sparsity, roughly half the possible connections should exist
        let synapses = random.synapse_count();
        let max_possible = 10 * 10;
        assert!(synapses > max_possible / 3); // At least some connections
        assert!(synapses < max_possible * 2 / 3); // But not too many
    }

    #[test]
    fn test_add_synapse() {
        let mut fc = FullyConnected::new(10, None, 1234, true);

        // Add excitatory synapse
        fc.add_synapse(0, 1, 1);
        assert_eq!(fc.adjacency_matrix()[[0, 1]], 1);

        // Add inhibitory synapse
        fc.add_synapse(2, 3, -1);
        assert_eq!(fc.adjacency_matrix()[[2, 3]], -1);
    }

    #[test]
    #[should_panic]
    fn test_add_synapse_invalid_polarity() {
        let mut fc = FullyConnected::new(10, None, 1234, true);
        fc.add_synapse(0, 1, 2); // Invalid polarity
    }

    #[test]
    #[should_panic]
    fn test_add_synapse_out_of_bounds() {
        let mut fc = FullyConnected::new(10, None, 1234, true);
        fc.add_synapse(0, 15, 1); // dest out of bounds
    }
}
