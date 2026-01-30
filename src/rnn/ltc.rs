//! Liquid Time-Constant (LTC) RNN Layer
//!
//! Full RNN layer that handles sequence processing, batching, and state management
//! for LTC (Liquid Time-Constant) cells.

use crate::cells::LSTMCell;
use crate::cells::LTCCell;
use crate::wirings::Wiring;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// LTC RNN Layer
///
/// A full RNN layer that processes sequences using LTC cells.
/// Supports batching, state management, mixed memory (LSTM), and variable timespans.
///
/// # Type Parameters
/// * `B` - The backend type
#[derive(Module, Debug)]
pub struct LTC<B: Backend> {
    /// The LTC cell for processing individual timesteps
    cell: LTCCell<B>,
    /// Optional LSTM cell for mixed memory mode
    #[module(skip)]
    lstm_cell: Option<LSTMCell<B>>,
    /// Input size (number of features)
    #[module(skip)]
    input_size: usize,
    /// State size (number of neurons)
    #[module(skip)]
    state_size: usize,
    /// Motor/output size (from wiring)
    #[module(skip)]
    motor_size: usize,
    /// Whether input is batch-first (batch, seq, features) vs (seq, batch, features)
    #[module(skip)]
    batch_first: bool,
    /// Whether to return full sequence or just last timestep
    #[module(skip)]
    return_sequences: bool,
    /// Whether to use mixed memory (LSTM augmentation)
    #[module(skip)]
    mixed_memory: bool,
}

impl<B: Backend> LTC<B> {
    /// Create a new LTC RNN layer with the given wiring
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `wiring` - Wiring configuration defining the network structure
    /// * `device` - Device to create the module on
    pub fn new(input_size: usize, wiring: impl Wiring, device: &B::Device) -> Self {
        let state_size = wiring.units();
        let motor_size = wiring.output_dim().unwrap_or(state_size);

        let cell = LTCCell::new(&wiring, Some(input_size), device);

        Self {
            cell,
            lstm_cell: None,
            input_size,
            state_size,
            motor_size,
            batch_first: true,
            return_sequences: true,
            mixed_memory: false,
        }
    }

    /// Set whether input is batch-first (default: true)
    ///
    /// When true: input shape is [batch, seq, features]
    /// When false: input shape is [seq, batch, features]
    pub fn with_batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
    }

    /// Set whether to return full sequences (default: true)
    ///
    /// When true: returns all timesteps [batch, seq, state_size]
    /// When false: returns only last timestep [batch, state_size]
    pub fn with_return_sequences(mut self, return_sequences: bool) -> Self {
        self.return_sequences = return_sequences;
        self
    }

    /// Enable or disable mixed memory mode (LSTM augmentation)
    ///
    /// When enabled, an LSTM cell processes the LTC output for better long-term memory.
    /// The LSTM cell is initialized when this is called with `true`.
    ///
    /// # Arguments
    /// * `mixed_memory` - Whether to enable mixed memory mode
    /// * `device` - Device to create the LSTM cell on (required when enabling)
    pub fn with_mixed_memory(mut self, mixed_memory: bool, device: &B::Device) -> Self {
        self.mixed_memory = mixed_memory;
        if mixed_memory && self.lstm_cell.is_none() {
            // Create LSTM cell: input_size -> state_size
            self.lstm_cell = Some(LSTMCell::new(self.input_size, self.state_size, device));
        }
        self
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get state size (number of neurons)
    pub fn state_size(&self) -> usize {
        self.state_size
    }

    /// Get motor/output size
    pub fn motor_size(&self) -> usize {
        self.motor_size
    }

    /// Forward pass through the LTC RNN layer
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape:
    ///   - 3D batched: [batch, seq, features] if batch_first=true
    ///   - 3D batched: [seq, batch, features] if batch_first=false
    ///   - 2D unbatched: [seq, features]
    /// * `state` - Optional initial state tensor of shape [batch, state_size]
    /// * `timespans` - Optional time intervals tensor of shape [batch, seq] or scalar
    ///
    /// # Returns
    /// Tuple of (output, final_state) where:
    /// - output: [batch, seq, motor_size] or [batch, motor_size] depending on return_sequences
    /// - final_state: [batch, state_size] or ([batch, state_size], [batch, state_size]) for mixed_memory
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: Option<Tensor<B, 2>>,
        timespans: Option<Tensor<B, 2>>,
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
            state.unwrap_or_else(|| Tensor::<B, 2>::zeros([batch_size, self.state_size], &device));

        // Default timespans (all ones)
        let timespans =
            timespans.unwrap_or_else(|| Tensor::<B, 2>::ones([batch_size, seq_len], &device));

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

            // Extract timespan for this timestep
            let step_time = timespans.clone().narrow(1, t, 1).squeeze(1);

            // Forward through LTC cell
            let (output, new_state) = self.cell.forward(step_input, current_state, step_time);
            current_state = new_state;

            if self.return_sequences {
                outputs.push(output);
            } else if t == seq_len - 1 {
                // Only keep last output if not returning sequences
                outputs.push(output);
            }
        }

        // Stack outputs into final tensor
        let output = Tensor::stack(outputs, 1); // [batch, seq, motor_size]
        (output, current_state)
    }

    /// Forward pass with mixed memory (LSTM augmentation)
    ///
    /// This follows the Python implementation order: LSTM first (for memory),
    /// then LTC (for continuous-time dynamics).
    ///
    /// This is only available when mixed_memory is enabled
    pub fn forward_mixed(
        &self,
        input: Tensor<B, 3>,
        state: Option<(Tensor<B, 2>, Tensor<B, 2>)>,
        timespans: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, (Tensor<B, 2>, Tensor<B, 2>))
    where
        B: Backend,
    {
        if !self.mixed_memory {
            panic!("Mixed memory not enabled. Call with_mixed_memory(true) first.");
        }

        let device = input.device();

        // Get dimensions
        let (batch_size, seq_len, _) = if self.batch_first {
            let dims = input.dims();
            (dims[0], dims[1], dims[2])
        } else {
            let dims = input.dims();
            (dims[1], dims[0], dims[2])
        };

        // Initialize states if not provided
        let (mut h_state, mut c_state) = state.unwrap_or_else(|| {
            (
                Tensor::<B, 2>::zeros([batch_size, self.state_size], &device),
                Tensor::<B, 2>::zeros([batch_size, self.state_size], &device),
            )
        });

        // Default timespans
        let timespans =
            timespans.unwrap_or_else(|| Tensor::<B, 2>::ones([batch_size, seq_len], &device));

        // Collect outputs
        let mut outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

        // Get LSTM cell reference (it should exist if mixed_memory is true)
        let lstm = self.lstm_cell.as_ref().expect("LSTM cell not initialized");

        for t in 0..seq_len {
            // Extract input for this timestep
            let step_input = if self.batch_first {
                input.clone().narrow(1, t, 1).squeeze(1)
            } else {
                input.clone().narrow(0, t, 1).squeeze(0)
            };

            // Extract timespan
            let step_time = timespans.clone().narrow(1, t, 1).squeeze(1);

            // FIRST: Forward through LSTM for memory (matches Python implementation)
            let (new_h, new_c) = lstm.forward(step_input.clone(), (h_state, c_state));
            h_state = new_h.clone();
            c_state = new_c;

            // SECOND: Forward through LTC cell with LSTM hidden state
            let (ltc_output, new_ltc_state) =
                self.cell.forward(step_input, h_state.clone(), step_time);
            h_state = new_ltc_state;

            if self.return_sequences {
                outputs.push(ltc_output);
            } else if t == seq_len - 1 {
                outputs.push(ltc_output);
            }
        }

        // Stack outputs
        let output = Tensor::stack(outputs, 1);
        (output, (h_state, c_state))
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
    fn test_ltc_rnn_creation() {
        let device = get_test_device();
        let wiring = FullyConnected::new(50, None, 1234, true);

        let ltc = LTC::<TestBackend>::new(20, wiring, &device);

        assert_eq!(ltc.input_size(), 20);
        assert_eq!(ltc.state_size(), 50);
    }

    #[test]
    fn test_ltc_rnn_forward_batch_first() {
        let device = get_test_device();
        let wiring = FullyConnected::new(50, None, 1234, true);
        let ltc = LTC::<TestBackend>::new(20, wiring, &device).with_batch_first(true);

        // [batch, seq, features]
        let input = Tensor::<TestBackend, 3>::zeros([4, 10, 20], &device);

        let (output, state) = ltc.forward(input, None, None);

        // [batch, seq, state_size]
        assert_eq!(output.dims(), [4, 10, 50]);
        assert_eq!(state.dims(), [4, 50]);
    }

    #[test]
    fn test_ltc_rnn_forward_seq_first() {
        let device = get_test_device();
        let wiring = FullyConnected::new(50, None, 1234, true);
        let ltc = LTC::<TestBackend>::new(20, wiring, &device).with_batch_first(false);

        // [seq, batch, features]
        let input = Tensor::<TestBackend, 3>::zeros([10, 4, 20], &device);

        let (output, state) = ltc.forward(input, None, None);

        // Output is always [batch, seq, state_size] for consistency
        assert_eq!(output.dims(), [4, 10, 50]);
    }

    #[test]
    fn test_ltc_rnn_return_last_only() {
        let device = get_test_device();
        let wiring = FullyConnected::new(50, None, 1234, true);
        let ltc = LTC::<TestBackend>::new(20, wiring, &device).with_return_sequences(false);

        let input = Tensor::<TestBackend, 3>::zeros([4, 10, 20], &device);

        let (output, state) = ltc.forward(input, None, None);

        // When return_sequences=false, we still return 3D with seq=1
        assert_eq!(output.dims(), [4, 1, 50]);
        assert_eq!(state.dims(), [4, 50]);
    }

    #[test]
    fn test_ltc_rnn_with_initial_state() {
        let device = get_test_device();
        let wiring = FullyConnected::new(50, None, 1234, true);
        let ltc = LTC::<TestBackend>::new(20, wiring, &device);

        let input = Tensor::<TestBackend, 3>::zeros([4, 10, 20], &device);
        let initial_state = Tensor::<TestBackend, 2>::ones([4, 50], &device);

        let (output, state) = ltc.forward(input, Some(initial_state), None);

        assert_eq!(output.dims(), [4, 10, 50]);
        assert_eq!(state.dims(), [4, 50]);
    }

    #[test]
    fn test_ltc_rnn_with_timespans() {
        let device = get_test_device();
        let wiring = FullyConnected::new(50, None, 1234, true);
        let ltc = LTC::<TestBackend>::new(20, wiring, &device);

        let input = Tensor::<TestBackend, 3>::zeros([4, 10, 20], &device);
        // Variable time intervals
        let timespans = Tensor::<TestBackend, 2>::full([4, 10], 0.5, &device);

        let (output, state) = ltc.forward(input, None, Some(timespans));

        assert_eq!(output.dims(), [4, 10, 50]);
        assert_eq!(state.dims(), [4, 50]);
    }

    #[test]
    fn test_ltc_rnn_with_ncp_wiring() {
        let device = get_test_device();
        let wiring = AutoNCP::new(64, 8, 0.5, 22222);
        let ltc = LTC::<TestBackend>::new(20, wiring, &device);

        let input = Tensor::<TestBackend, 3>::zeros([2, 5, 20], &device);
        let (output, state) = ltc.forward(input, None, None);

        // Output should be motor_size (8)
        assert_eq!(output.dims(), [2, 5, 8]);
        assert_eq!(state.dims(), [2, 64]);
    }

    #[test]
    fn test_ltc_rnn_sequence_processing() {
        let device = get_test_device();
        let wiring = FullyConnected::new(20, None, 1234, true);
        let ltc = LTC::<TestBackend>::new(10, wiring, &device);

        // Test different sequence lengths
        for seq_len in [1, 5, 20] {
            let input = Tensor::<TestBackend, 3>::zeros([2, seq_len, 10], &device);
            let (output, _) = ltc.forward(input, None, None);

            assert_eq!(output.dims(), [2, seq_len, 20]);
        }
    }
}
