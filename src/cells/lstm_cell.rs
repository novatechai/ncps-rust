use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Standard LSTM cell for "mixed_memory" mode
///
/// Implements the standard LSTM equations:
/// - i = tanh(W_ii @ x + b_ii + W_hi @ h + b_hi)
/// - g = sigmoid(W_ig @ x + b_ig + W_hg @ h + b_hg)
/// - f = sigmoid(W_if @ x + b_if + W_hf @ h + b_hf + 1)
/// - o = sigmoid(W_io @ x + b_io + W_ho @ h + b_ho)
/// - c' = f * c + i * g
/// - h' = o * tanh(c')
#[derive(Module, Debug)]
pub struct LSTMCell<B: Backend> {
    input_size: usize,
    hidden_size: usize,
    input_map: Linear<B>,     // Maps input to 4 * hidden_size (with bias)
    recurrent_map: Linear<B>, // Maps hidden state to 4 * hidden_size (no bias)
}

impl<B: Backend> LSTMCell<B> {
    /// Create a new LSTM cell
    ///
    /// # Arguments
    /// * `input_size` - Size of the input features
    /// * `hidden_size` - Size of the hidden state
    /// * `device` - Device to create the module on
    ///
    /// # Returns
    /// A new LSTMCell instance with initialized weights
    pub fn new(input_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        let input_map = LinearConfig::new(input_size, 4 * hidden_size)
            .with_bias(true)
            .init(device);

        let recurrent_map = LinearConfig::new(hidden_size, 4 * hidden_size)
            .with_bias(false)
            .init(device);

        Self {
            input_size,
            hidden_size,
            input_map,
            recurrent_map,
        }
    }

    /// Get the input size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get the hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Perform a forward pass through the LSTM cell
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `[batch_size, input_size]`
    /// * `states` - Tuple of (hidden_state, cell_state), each of shape `[batch_size, hidden_size]`
    ///
    /// # Returns
    /// Tuple of (new_hidden_state, new_cell_state)
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        states: (Tensor<B, 2>, Tensor<B, 2>),
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (hidden_state, cell_state) = states;

        // Compute combined transformation
        let input_contrib = self.input_map.forward(input);
        let recurrent_contrib = self.recurrent_map.forward(hidden_state.clone());
        let z = input_contrib + recurrent_contrib;

        // Split into 4 gates
        let chunks = z.chunk(4, 1);
        let input_activation = chunks[0].clone(); // i
        let input_gate = chunks[1].clone(); // ig
        let forget_gate = chunks[2].clone(); // fg
        let output_gate = chunks[3].clone(); // og

        // Apply activations
        let input_activation = input_activation.tanh();
        let input_gate = activation::sigmoid(input_gate);
        let forget_gate = activation::sigmoid(forget_gate + 1.0); // Add 1.0 bias to forget gate
        let output_gate = activation::sigmoid(output_gate);

        // Update cell state: c' = f * c + i * g
        let new_cell = cell_state * forget_gate + input_activation * input_gate;

        // Update hidden state: h' = o * tanh(c')
        let new_hidden = new_cell.clone().tanh() * output_gate;

        (new_hidden, new_cell)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::backend::Backend as BurnBackend;

    type TestBackend = NdArray<f32>;
    type TestDevice = <TestBackend as BurnBackend>::Device;

    fn get_test_device() -> TestDevice {
        Default::default()
    }

    #[test]
    fn test_lstm_cell_creation() {
        let device = get_test_device();
        let cell = LSTMCell::<TestBackend>::new(20, 50, &device);

        assert_eq!(cell.input_size(), 20);
        assert_eq!(cell.hidden_size(), 50);
    }

    #[test]
    fn test_lstm_forward() {
        let device = get_test_device();
        let cell = LSTMCell::<TestBackend>::new(20, 50, &device);

        let batch_size = 4;
        let input = Tensor::<TestBackend, 2>::zeros([batch_size, 20], &device);
        let h = Tensor::<TestBackend, 2>::zeros([batch_size, 50], &device);
        let c = Tensor::<TestBackend, 2>::zeros([batch_size, 50], &device);

        let (new_h, new_c) = cell.forward(input, (h, c));

        assert_eq!(new_h.dims(), [batch_size, 50]);
        assert_eq!(new_c.dims(), [batch_size, 50]);
    }

    #[test]
    fn test_lstm_state_persistence() {
        let device = get_test_device();
        let cell = LSTMCell::<TestBackend>::new(10, 20, &device);

        // Sequence of inputs
        let input1 = Tensor::<TestBackend, 2>::random(
            [1, 10],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let input2 = Tensor::<TestBackend, 2>::random(
            [1, 10],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let input3 = Tensor::<TestBackend, 2>::random(
            [1, 10],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let mut h = Tensor::<TestBackend, 2>::zeros([1, 20], &device);
        let mut c = Tensor::<TestBackend, 2>::zeros([1, 20], &device);

        (h, c) = cell.forward(input1, (h, c));
        (h, c) = cell.forward(input2, (h, c));
        (h, c) = cell.forward(input3, (h, c));

        // States should have evolved
        let h_sum = h.sum().into_scalar();
        let c_sum = c.sum().into_scalar();
        assert!(
            h_sum != 0.0 || c_sum != 0.0,
            "States should have changed after processing sequence"
        );
    }

    #[test]
    fn test_lstm_forget_gate() {
        let device = get_test_device();
        let cell = LSTMCell::<TestBackend>::new(10, 20, &device);

        // Start with cell state at value 10
        let h = Tensor::<TestBackend, 2>::zeros([1, 20], &device);
        let c = Tensor::<TestBackend, 2>::ones([1, 20], &device) * 10.0;
        let input = Tensor::<TestBackend, 2>::zeros([1, 10], &device);

        let (_, new_c) = cell.forward(input, (h, c));

        // Cell state should have changed (forget gate should modify it)
        let c_sum_old = 10.0 * 20.0;
        let c_sum_new: f32 = new_c.sum().into_scalar();
        assert!(
            (c_sum_new - c_sum_old).abs() > 0.1,
            "Forget gate should modify cell state"
        );
    }

    #[test]
    fn test_lstm_batch_sizes() {
        let device = get_test_device();
        let cell = LSTMCell::<TestBackend>::new(20, 50, &device);

        for batch_size in [1, 4, 16, 32] {
            let input = Tensor::<TestBackend, 2>::zeros([batch_size, 20], &device);
            let h = Tensor::<TestBackend, 2>::zeros([batch_size, 50], &device);
            let c = Tensor::<TestBackend, 2>::zeros([batch_size, 50], &device);

            let (new_h, new_c) = cell.forward(input, (h, c));

            assert_eq!(new_h.dims(), [batch_size, 50]);
            assert_eq!(new_c.dims(), [batch_size, 50]);
        }
    }
}
