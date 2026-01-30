//! Wired CfC Cell Implementation
//!
//! Multi-layer CfC cell that respects NCP wiring structure.
//! Creates separate CfC cells for each layer of the wiring, following the
//! connectivity patterns defined by the adjacency matrices.

use crate::cells::{CfCCell, CfcMode};
use crate::wirings::Wiring;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ndarray::Array2;

/// Wired CfC Cell - Multi-layer CfC respecting NCP wiring structure
///
/// This cell creates separate CfC cells for each layer of the wiring,
/// with appropriate sparsity masks derived from the adjacency matrices.
#[derive(Module, Debug)]
pub struct WiredCfCCell<B: Backend> {
    /// The layers of CfC cells, one per wiring layer
    #[module(child_list)]
    layers: Vec<CfCCell<B>>,
    /// Total number of neurons (state size)
    #[module(skip)]
    state_size: usize,
    /// Motor (output) size
    #[module(skip)]
    motor_size: usize,
    /// Sensory (input) size
    #[module(skip)]
    sensory_size: usize,
    /// Layer sizes for state partitioning
    #[module(skip)]
    layer_sizes: Vec<usize>,
}

impl<B: Backend> WiredCfCCell<B> {
    /// Create a new WiredCfCCell with a given wiring
    ///
    /// # Arguments
    /// * `wiring` - The wiring configuration (must be built)
    /// * `device` - The device for tensor operations
    /// * `mode` - The CfC operating mode (default, pure, or no_gate)
    pub fn new(wiring: &dyn Wiring, device: &B::Device, mode: CfcMode) -> Self {
        if !wiring.is_built() {
            panic!(
                "Wiring error! Unknown number of input features. \
                 Please build the wiring first by calling wiring.build(input_size)."
            );
        }

        let num_layers = wiring.num_layers();
        let input_dim = wiring.input_dim().unwrap();
        let state_size = wiring.units();
        let motor_size = wiring.output_dim().unwrap_or(state_size);
        let mut layers: Vec<CfCCell<B>> = Vec::with_capacity(num_layers);

        // For fully_connected-like case, we create one CfC cell per layer
        // Each layer's CfC takes as input: previous layer output (or sensory input)
        // and its own hidden state (which is the layer's neurons)

        // For the wiring-based CfC, we need to create sparsity masks that constrain
        // the connections based on the wiring adjacency matrix

        for l in 0..num_layers {
            let hidden_units = wiring.get_neurons_of_layer(l);
            let num_hidden = hidden_units.len();

            // The input to this CfC cell is:
            // - For layer 0: sensory inputs + layer 0 hidden state
            // - For layer N: layer N-1 output + layer N hidden state
            // But CfCCell already handles concatenation internally, so we just need
            // to tell it the input size (previous layer output size or sensory size)
            let prev_layer_size = if l == 0 {
                input_dim
            } else {
                wiring.get_neurons_of_layer(l - 1).len()
            };

            // Build input sparsity mask based on wiring connections
            // The mask should have shape: [prev_layer_size, num_hidden]
            // Extended with identity for recurrent connections: [prev_layer_size + num_hidden, num_hidden]
            let input_sparsity = if l == 0 {
                // First layer: use sensory adjacency matrix
                let sensory_matrix = wiring
                    .sensory_adjacency_matrix()
                    .expect("Sensory adjacency matrix required for first layer");
                // Extract columns for this layer's neurons
                let mut mask = Array2::zeros((input_dim, num_hidden));
                for (i, &neuron_id) in hidden_units.iter().enumerate() {
                    let col = sensory_matrix.column(neuron_id);
                    for (row, &val) in col.iter().enumerate() {
                        mask[[row, i]] = val.abs() as f32;
                    }
                }
                mask
            } else {
                // Subsequent layers: use adjacency matrix from previous layer
                let adj_matrix = wiring.adjacency_matrix();
                let prev_layer_neurons = wiring.get_neurons_of_layer(l - 1);

                // Create mask: [prev_layer_size x current_layer_size]
                let mut mask = Array2::zeros((prev_layer_neurons.len(), num_hidden));
                for (i, &current_neuron) in hidden_units.iter().enumerate() {
                    for (j, &prev_neuron) in prev_layer_neurons.iter().enumerate() {
                        mask[[j, i]] = adj_matrix[[prev_neuron, current_neuron]].abs() as f32;
                    }
                }
                mask
            };

            // Extend mask with identity matrix for recurrent connections
            // The extended mask should be: [prev_layer_size + num_hidden, num_hidden]
            let mut extended_mask =
                Array2::zeros((input_sparsity.nrows() + num_hidden, num_hidden));
            // Copy input sparsity to top portion
            for i in 0..input_sparsity.nrows() {
                for j in 0..num_hidden {
                    extended_mask[[i, j]] = input_sparsity[[i, j]];
                }
            }
            // Set identity matrix in bottom portion (recurrent connections)
            for i in 0..num_hidden {
                extended_mask[[input_sparsity.nrows() + i, i]] = 1.0;
            }

            // Create CfC cell:
            // - input_size should be the previous layer's output dimension (or sensory size)
            // - hidden_size is this layer's number of neurons
            // CfCCell will internally concatenate [input, hx], so it creates
            // a Linear layer with input_size=input_size+hidden_size
            let cell = CfCCell::new(prev_layer_size, num_hidden, device)
                .with_mode(mode)
                .with_sparsity_mask(extended_mask, device);

            layers.push(cell);
        }

        let layer_sizes: Vec<usize> = (0..num_layers)
            .map(|l| wiring.get_neurons_of_layer(l).len())
            .collect();

        Self {
            layers,
            state_size,
            motor_size,
            sensory_size: input_dim,
            layer_sizes,
        }
    }

    /// Create a new WiredCfCCell with default mode
    pub fn with_default_mode(wiring: &dyn Wiring, device: &B::Device) -> Self {
        Self::new(wiring, device, CfcMode::Default)
    }

    /// Get the total state size (sum of all layer neurons)
    pub fn state_size(&self) -> usize {
        self.state_size
    }

    /// Get the motor (output) size
    pub fn motor_size(&self) -> usize {
        self.motor_size
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layer_sizes.len()
    }

    /// Get the sizes of each layer
    pub fn layer_sizes(&self) -> &[usize] {
        &self.layer_sizes
    }

    /// Get the sensory (input) size
    pub fn sensory_size(&self) -> usize {
        self.sensory_size
    }

    /// Get the output size (alias for motor_size)
    pub fn output_size(&self) -> usize {
        self.motor_size()
    }

    /// Perform a forward pass through the wired CfC cell
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, sensory_size]
    /// * `hx` - Hidden state tensor of shape [batch_size, state_size]
    /// * `ts` - Time step (scalar)
    ///
    /// # Returns
    /// * `(output, new_hidden)` - Output is motor neurons, new_hidden is full state
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        hx: Tensor<B, 2>,
        ts: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Split hx into layer states using narrow
        let mut h_states: Vec<Tensor<B, 2>> = Vec::with_capacity(self.num_layers());
        let mut start_idx = 0;

        for &layer_size in &self.layer_sizes {
            // Use narrow to extract slice [batch, start:start+size]
            let layer_state = hx.clone().narrow(1, start_idx, layer_size);
            h_states.push(layer_state);
            start_idx += layer_size;
        }

        // Forward through each layer
        let mut new_h_states: Vec<Tensor<B, 2>> = Vec::with_capacity(self.num_layers());
        let mut layer_input = input;

        for (i, layer) in self.layers.iter().enumerate() {
            let h_state = h_states[i].clone();
            let (new_h, _) = layer.forward(layer_input, h_state, ts);
            layer_input = new_h.clone();
            new_h_states.push(new_h);
        }

        // Concatenate new states
        let new_hx = Tensor::cat(new_h_states, 1);

        // For FullyConnected wiring: output_dim might differ from units
        // The output neurons are the first motor_size neurons of the state
        // For NCP wiring: motor neurons are naturally the first layer
        let output = if self.motor_size != self.state_size {
            // Need to narrow to get only motor neurons
            layer_input.narrow(1, 0, self.motor_size)
        } else {
            layer_input
        };

        (output, new_hx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wirings::{AutoNCP, FullyConnected, NCP};
    use burn::backend::NdArray;
    use burn::tensor::backend::Backend as BurnBackend;

    type TestBackend = NdArray<f32>;
    type TestDevice = <TestBackend as BurnBackend>::Device;

    fn get_test_device() -> TestDevice {
        Default::default()
    }

    fn create_wired_cell_with_ncp() -> WiredCfCCell<TestBackend> {
        let device = get_test_device();
        let mut wiring = AutoNCP::new(32, 8, 0.5, 22222);
        wiring.build(16);
        WiredCfCCell::new(&wiring, &device, CfcMode::Default)
    }

    #[test]
    fn test_wired_cfc_creation() {
        let cell = create_wired_cell_with_ncp();

        assert_eq!(cell.state_size(), 32);
        assert_eq!(cell.motor_size(), 8);
        assert_eq!(cell.num_layers(), 3);
    }

    #[test]
    fn test_wired_cfc_layer_sizes() {
        let cell = create_wired_cell_with_ncp();
        let sizes = cell.layer_sizes();

        // Should have 3 layers
        assert_eq!(sizes.len(), 3);
        // Total should equal state_size
        let total: usize = sizes.iter().sum();
        assert_eq!(total, cell.state_size());
    }

    #[test]
    fn test_wired_cfc_forward() {
        let device = get_test_device();
        let cell = create_wired_cell_with_ncp();

        let batch_size = 4;
        let input = Tensor::<TestBackend, 2>::zeros([batch_size, 16], &device);
        let hx = Tensor::<TestBackend, 2>::zeros([batch_size, 32], &device);

        let (output, new_hidden) = cell.forward(input, hx, 1.0);

        // Output should be motor_size
        assert_eq!(output.dims(), [batch_size, 8]);
        // New hidden should preserve full state
        assert_eq!(new_hidden.dims(), [batch_size, 32]);
    }

    #[test]
    fn test_wired_cfc_state_partitioning() {
        let device = get_test_device();
        let cell = create_wired_cell_with_ncp();

        // Create state with different values for each layer
        let layer_sizes = cell.layer_sizes().to_vec();
        let hx_parts: Vec<Tensor<TestBackend, 2>> = layer_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| Tensor::<TestBackend, 2>::full([2, size], (i + 1) as f32, &device))
            .collect();

        let hx = Tensor::cat(hx_parts, 1);

        let input = Tensor::<TestBackend, 2>::zeros([2, 16], &device);
        let (output, new_hidden) = cell.forward(input, hx, 1.0);

        // Verify state was processed correctly
        assert_eq!(new_hidden.dims(), [2, 32]);
        assert_eq!(output.dims(), [2, 8]);
    }

    #[test]
    fn test_wired_cfc_with_different_wirings() {
        let device = get_test_device();

        // Test with manually configured NCP
        let mut wiring = NCP::new(10, 8, 5, 6, 6, 4, 6, 22222);
        wiring.build(10);
        let cell = WiredCfCCell::<TestBackend>::new(&wiring, &device, CfcMode::Default);

        assert_eq!(cell.state_size(), 23); // 10 + 8 + 5
        assert_eq!(cell.num_layers(), 3);
    }

    #[test]
    fn test_wired_cfc_information_flow() {
        let device = get_test_device();
        let cell = create_wired_cell_with_ncp();

        // Test that information flows from sensory through all layers
        let input1 = Tensor::<TestBackend, 2>::zeros([1, 16], &device);
        let input2 = Tensor::<TestBackend, 2>::ones([1, 16], &device);
        let hx = Tensor::<TestBackend, 2>::zeros([1, 32], &device);

        let (out1, _) = cell.forward(input1, hx.clone(), 1.0);
        let (out2, _) = cell.forward(input2, hx, 1.0);

        let diff = (out1 - out2).abs().sum().into_scalar();
        assert!(
            diff > 0.0,
            "Different inputs should produce different outputs"
        );
    }

    #[test]
    fn test_wired_cfc_with_fully_connected() {
        let device = get_test_device();

        // Test with FullyConnected wiring
        let mut wiring = FullyConnected::new(20, Some(5), 1234, true);
        wiring.build(10);
        let cell = WiredCfCCell::<TestBackend>::new(&wiring, &device, CfcMode::Default);

        assert_eq!(cell.state_size(), 20);
        assert_eq!(cell.motor_size(), 5);
        // FullyConnected has only 1 layer
        assert_eq!(cell.num_layers(), 1);

        // Test forward pass
        let input = Tensor::<TestBackend, 2>::zeros([2, 10], &device);
        let hx = Tensor::<TestBackend, 2>::zeros([2, 20], &device);
        let (output, new_hidden) = cell.forward(input, hx, 1.0);

        assert_eq!(output.dims(), [2, 5]);
        assert_eq!(new_hidden.dims(), [2, 20]);
    }
}
