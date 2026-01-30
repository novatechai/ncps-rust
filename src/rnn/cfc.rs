//! Closed-form Continuous-time (CfC) RNN Layer
//!
//! Full RNN layer that handles sequence processing, batching, and state management
//! for CfC (Closed-form Continuous-time) cells.

use crate::cells::CfCCell;
use crate::wirings::Wiring;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// CfC RNN Layer
///
/// A full RNN layer that processes sequences using CfC cells.
/// Supports batching, state management, different CfC modes, and optional projections.
///
/// # Type Parameters
/// * `B` - The backend type
#[derive(Module, Debug)]
pub struct CfC<B: Backend> {
    /// The CfC cell for processing individual timesteps
    cell: CfCCell<B>,
    /// Optional projection layer
    proj: Option<Linear<B>>,
    /// Input size (number of features)
    #[module(skip)]
    input_size: usize,
    /// Hidden/output size
    #[module(skip)]
    hidden_size: usize,
    /// Whether input is batch-first
    #[module(skip)]
    batch_first: bool,
    /// Whether to return full sequence or just last timestep
    #[module(skip)]
    return_sequences: bool,
    /// Projection size (if using NCP wiring)
    #[module(skip)]
    proj_size: Option<usize>,
    /// Output size (hidden_size or proj_size)
    #[module(skip)]
    output_size: usize,
}

impl<B: Backend> CfC<B> {
    /// Create a new CfC RNN layer with simple hidden size
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of hidden units
    /// * `device` - Device to create the module on
    pub fn new(input_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        let cell = CfCCell::new(input_size, hidden_size, device);

        Self {
            cell,
            proj: None,
            input_size,
            hidden_size,
            batch_first: true,
            return_sequences: true,
            proj_size: None,
            output_size: hidden_size,
        }
    }

    /// Create a new CfC RNN layer with wiring configuration
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `wiring` - Wiring configuration (e.g., AutoNCP)
    /// * `device` - Device to create the module on
    pub fn with_wiring(input_size: usize, wiring: impl Wiring, device: &B::Device) -> Self {
        let state_size = wiring.units();
        let motor_size = wiring.output_dim().unwrap_or(state_size);

        let cell = CfCCell::new(input_size, state_size, device);

        let output_size = motor_size;

        // Create projection layer if motor_size differs from state_size
        let proj = if motor_size != state_size {
            Some(
                LinearConfig::new(state_size, motor_size)
                    .with_bias(true)
                    .init(device),
            )
        } else {
            None
        };

        Self {
            cell,
            proj,
            input_size,
            hidden_size: state_size,
            batch_first: true,
            return_sequences: true,
            proj_size: if motor_size != state_size {
                Some(motor_size)
            } else {
                None
            },
            output_size,
        }
    }

    /// Set whether input is batch-first (default: true)
    pub fn with_batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
    }

    /// Set whether to return full sequences (default: true)
    pub fn with_return_sequences(mut self, return_sequences: bool) -> Self {
        self.return_sequences = return_sequences;
        self
    }

    /// Set projection size for motor outputs and create the projection layer
    pub fn with_proj_size(mut self, proj_size: usize) -> Self {
        let device = self.get_device();
        self.proj = Some(
            LinearConfig::new(self.hidden_size, proj_size)
                .with_bias(true)
                .init(&device),
        );
        self.proj_size = Some(proj_size);
        self.output_size = proj_size;
        self
    }

    /// Configure backbone - currently a no-op for API compatibility
    pub fn with_backbone(self, _units: usize, _layers: usize, _dropout: f64) -> Self {
        self
    }

    /// Helper method to get the device from the cell (defaults to CPU)
    fn get_device(&self) -> B::Device {
        B::Device::default()
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get output size (considering projection)
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Forward pass through the CfC RNN layer
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape:
    ///   - 3D: [batch, seq, features] if batch_first=true
    ///   - 3D: [seq, batch, features] if batch_first=false
    /// * `state` - Optional initial hidden state tensor of shape [batch, hidden_size]
    /// * `timespans` - Optional time intervals (scalar used for all timesteps if None)
    ///
    /// # Returns
    /// Tuple of (output, final_state) where:
    /// - output: [batch, seq, output_size] or [batch, output_size] depending on return_sequences
    /// - final_state: [batch, hidden_size]
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: Option<Tensor<B, 2>>,
        _timespans: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let device = input.device();

        // Get dimensions
        let (batch_size, seq_len, _) = if self.batch_first {
            let dims = input.dims();
            (dims[0], dims[1], dims[2])
        } else {
            let dims = input.dims();
            (dims[1], dims[0], dims[2])
        };

        // Initialize state if not provided
        let mut current_state =
            state.unwrap_or_else(|| Tensor::<B, 2>::zeros([batch_size, self.hidden_size], &device));

        // Collect outputs for each timestep
        let mut outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Extract input for this timestep
            let step_input = if self.batch_first {
                // input[batch, t, features] -> [batch, features]
                input.clone().narrow(1, t, 1).squeeze(1)
            } else {
                // input[t, batch, features] -> [batch, features]
                input.clone().narrow(0, t, 1).squeeze(0)
            };

            // Forward through CfC cell (ts defaults to 1.0)
            let (mut output, new_state) = self.cell.forward(step_input, current_state, 1.0);
            current_state = new_state;

            // Apply projection if configured
            if let Some(ref proj) = self.proj {
                output = proj.forward(output);
            }

            if self.return_sequences {
                outputs.push(output);
            } else if t == seq_len - 1 {
                // Only keep last output if not returning sequences
                outputs.push(output);
            }
        }

        // Stack outputs into final tensor
        let output = Tensor::stack(outputs, 1); // [batch, seq, output_size]
        (output, current_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wirings::{AutoNCP, FullyConnected};
    use burn::backend::NdArray;
    use burn::tensor::backend::Backend as BurnBackend;

    type TestBackend = NdArray<f32>;
    type TestDevice = <TestBackend as BurnBackend>::Device;

    fn get_test_device() -> TestDevice {
        Default::default()
    }

    #[test]
    fn test_cfc_rnn_creation() {
        let device = get_test_device();
        let cfc = CfC::<TestBackend>::new(20, 50, &device);

        assert_eq!(cfc.input_size(), 20);
        assert_eq!(cfc.hidden_size(), 50);
        assert_eq!(cfc.output_size(), 50);
    }

    #[test]
    fn test_cfc_rnn_with_wiring() {
        let device = get_test_device();
        let wiring = AutoNCP::new(32, 8, 0.5, 22222);
        let cfc = CfC::<TestBackend>::with_wiring(20, wiring, &device);

        assert_eq!(cfc.output_size(), 8);
    }

    #[test]
    fn test_cfc_rnn_forward() {
        let device = get_test_device();
        let cfc = CfC::<TestBackend>::new(20, 50, &device);

        let input = Tensor::<TestBackend, 3>::zeros([4, 10, 20], &device);
        let (output, state) = cfc.forward(input, None, None);

        assert_eq!(output.dims(), [4, 10, 50]);
        assert_eq!(state.dims(), [4, 50]);
    }

    #[test]
    fn test_cfc_rnn_with_projection() {
        let device = get_test_device();
        let cfc = CfC::<TestBackend>::new(20, 50, &device).with_proj_size(10);

        let input = Tensor::<TestBackend, 3>::zeros([4, 10, 20], &device);
        let (output, _) = cfc.forward(input, None, None);

        // Output should be projected to 10
        assert_eq!(output.dims(), [4, 10, 10]);
        assert_eq!(cfc.output_size(), 10);
    }

    #[test]
    fn test_cfc_rnn_backbone_config() {
        let device = get_test_device();
        let cfc = CfC::<TestBackend>::new(20, 50, &device).with_backbone(128, 2, 0.1);

        let input = Tensor::<TestBackend, 3>::zeros([2, 5, 20], &device);
        let (output, _) = cfc.forward(input, None, None);

        assert_eq!(output.dims(), [2, 5, 50]);
    }

    #[test]
    fn test_cfc_rnn_return_last_only() {
        let device = get_test_device();
        let cfc = CfC::<TestBackend>::new(20, 50, &device).with_return_sequences(false);

        let input = Tensor::<TestBackend, 3>::zeros([4, 10, 20], &device);
        let (output, state) = cfc.forward(input, None, None);

        // Should return [batch, 1, hidden_size]
        assert_eq!(output.dims(), [4, 1, 50]);
        assert_eq!(state.dims(), [4, 50]);
    }

    #[test]
    fn test_cfc_rnn_seq_first() {
        let device = get_test_device();
        let cfc = CfC::<TestBackend>::new(20, 50, &device).with_batch_first(false);

        // [seq, batch, features]
        let input = Tensor::<TestBackend, 3>::zeros([10, 4, 20], &device);
        let (output, state) = cfc.forward(input, None, None);

        assert_eq!(output.dims(), [4, 10, 50]);
        assert_eq!(state.dims(), [4, 50]);
    }

    #[test]
    fn test_cfc_rnn_with_initial_state() {
        let device = get_test_device();
        let cfc = CfC::<TestBackend>::new(20, 50, &device);

        let input = Tensor::<TestBackend, 3>::zeros([4, 10, 20], &device);
        let initial_state = Tensor::<TestBackend, 2>::ones([4, 50], &device);

        let (output, state) = cfc.forward(input, Some(initial_state), None);

        assert_eq!(output.dims(), [4, 10, 50]);
        assert_eq!(state.dims(), [4, 50]);
    }
}
