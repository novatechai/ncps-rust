//! Closed-form Continuous-time (CfC) Cell Implementation
//!
//! The CfC cell is a fast approximation of the LTC (Liquid Time-Constant) cell.
//! It provides closed-form solutions to continuous-time neural dynamics without
//! requiring iterative ODE solvers.
//!
//! Three modes are supported:
//! - **Default**: Gated interpolation between two feedforward paths
//! - **Pure**: Direct ODE solution without gating
//! - **NoGate**: Simplified gating with addition instead of interpolation

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ndarray::Array2;

/// CfC cell operating modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CfcMode {
    /// Default gated mode: h = tanh(ff1) * (1 - σ) + tanh(ff2) * σ
    Default = 0,
    /// Pure ODE solution without gating
    Pure = 1,
    /// No-gate mode: h = ff1 + tanh(ff2) * σ
    NoGate = 2,
}

/// A Closed-form Continuous-time cell
///
/// This is an RNNCell that processes single time-steps. To get a full RNN
/// that can process sequences, see the full RNN layer implementation.
///
/// # Type Parameters
/// * `B` - The backend type
#[derive(Module, Debug)]
pub struct CfCCell<B: Backend> {
    #[module(skip)]
    input_size: usize,
    #[module(skip)]
    hidden_size: usize,
    /// Mode: 0=Default, 1=Pure, 2=NoGate
    #[module(skip)]
    mode: u8,
    /// Whether sparsity mask is enabled
    #[module(skip)]
    has_sparsity_mask: bool,
    ff1: Linear<B>,
    ff2: Option<Linear<B>>,
    time_a: Option<Linear<B>>,
    time_b: Option<Linear<B>>,
    w_tau: Option<Linear<B>>,
    a: Option<Linear<B>>,
    /// Sparsity mask for output (transposed from input mask)
    sparsity_mask: Option<Param<Tensor<B, 2>>>,
}

impl<B: Backend> CfCCell<B> {
    /// Create a new CfC cell
    pub fn new(input_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        let ff1 = LinearConfig::new(input_size + hidden_size, hidden_size)
            .with_bias(true)
            .init(device);

        let ff2 = LinearConfig::new(input_size + hidden_size, hidden_size)
            .with_bias(true)
            .init(device);

        let time_a = LinearConfig::new(input_size + hidden_size, hidden_size)
            .with_bias(true)
            .init(device);

        let time_b = LinearConfig::new(input_size + hidden_size, hidden_size)
            .with_bias(true)
            .init(device);

        Self {
            input_size,
            hidden_size,
            mode: 0, // Default
            has_sparsity_mask: false,
            ff1,
            ff2: Some(ff2),
            time_a: Some(time_a),
            time_b: Some(time_b),
            w_tau: None,
            a: None,
            sparsity_mask: None,
        }
    }

    /// Set the CfC mode (Default, Pure, or NoGate)
    pub fn with_mode(mut self, mode: CfcMode) -> Self {
        self.mode = match mode {
            CfcMode::Default => 0,
            CfcMode::Pure => 1,
            CfcMode::NoGate => 2,
        };
        self.reconfigure_for_mode();
        self
    }

    /// Configure backbone (currently a no-op, kept for API compatibility)
    pub fn with_backbone(self, _units: usize, _layers: usize, _dropout: f64) -> Self {
        // Backbone support would require dynamic layer sizing
        // For now, we keep the simple version working
        self
    }

    /// Set the backbone activation (currently a no-op, kept for API compatibility)
    pub fn with_activation(self, activation: &str) -> Self {
        let valid_activations = ["relu", "tanh", "gelu", "silu", "lecun_tanh"];
        if !valid_activations.contains(&activation) {
            panic!(
                "Unknown activation: {}. Valid options are {:?}",
                activation, valid_activations
            );
        }
        self
    }

    /// Set a sparsity mask to enforce wiring connectivity
    ///
    /// The mask should have shape [hidden_size, hidden_size] and contain
    /// 0s for blocked connections and 1s for allowed connections.
    /// Note: The mask is transposed internally to match PyTorch convention.
    pub fn with_sparsity_mask(mut self, mask: Array2<f32>, device: &B::Device) -> Self {
        let shape = mask.shape();
        // Transpose the mask to match PyTorch's convention (sparsity_mask.T)
        let transposed = mask.t();
        let data: Vec<f32> = transposed.iter().map(|&x| x.abs()).collect();
        let tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([shape[1], shape[0]]);
        self.sparsity_mask = Some(Param::from_tensor(tensor));
        self.has_sparsity_mask = true;
        self
    }

    /// Create a CfC cell from a wiring configuration
    pub fn from_wiring(
        input_size: usize,
        wiring: &dyn crate::wirings::Wiring,
        device: &B::Device,
    ) -> Self {
        let hidden_size = wiring.units();
        let mut cell = Self::new(input_size, hidden_size, device);

        // Apply sparsity mask from adjacency matrix
        let adj_matrix = wiring.adjacency_matrix();
        let shape = adj_matrix.shape();
        let data: Vec<f32> = adj_matrix.iter().map(|&x| x.abs() as f32).collect();
        let mask_tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([shape[0], shape[1]]);
        cell.sparsity_mask = Some(Param::from_tensor(mask_tensor));
        cell.has_sparsity_mask = true;

        cell
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get current mode
    pub fn mode(&self) -> CfcMode {
        match self.mode {
            0 => CfcMode::Default,
            1 => CfcMode::Pure,
            2 => CfcMode::NoGate,
            _ => CfcMode::Default,
        }
    }

    fn reconfigure_for_mode(&mut self) {
        let device = self.ff1.weight.device();

        match self.mode {
            1 => {
                // Pure mode: use w_tau and a, remove ff2/time parameters
                self.ff2 = None;
                self.time_a = None;
                self.time_b = None;

                self.w_tau = Some(
                    LinearConfig::new(1, self.hidden_size)
                        .with_bias(false)
                        .init(&device),
                );
                self.a = Some(
                    LinearConfig::new(1, self.hidden_size)
                        .with_bias(false)
                        .init(&device),
                );
            }
            _ => {
                // Default/NoGate mode: ensure ff2, time_a, time_b exist
                if self.ff2.is_none() {
                    self.ff2 = Some(
                        LinearConfig::new(self.input_size + self.hidden_size, self.hidden_size)
                            .with_bias(true)
                            .init(&device),
                    );
                }
                if self.time_a.is_none() {
                    self.time_a = Some(
                        LinearConfig::new(self.input_size + self.hidden_size, self.hidden_size)
                            .with_bias(true)
                            .init(&device),
                    );
                }
                if self.time_b.is_none() {
                    self.time_b = Some(
                        LinearConfig::new(self.input_size + self.hidden_size, self.hidden_size)
                            .with_bias(true)
                            .init(&device),
                    );
                }
                self.w_tau = None;
                self.a = None;
            }
        }
        // Note: sparsity_mask is preserved across mode changes
    }

    /// Check if this cell has a sparsity mask
    pub fn has_sparsity_mask(&self) -> bool {
        self.has_sparsity_mask
    }

    /// Apply sparsity mask to a tensor if mask exists
    fn apply_sparsity_mask(&self, tensor: Tensor<B, 2>) -> Tensor<B, 2> {
        if let Some(ref mask) = self.sparsity_mask {
            // Mask shape is [hidden_size, hidden_size], we need to broadcast
            // For output masking, we just multiply element-wise with the diagonal
            // or apply the full mask if needed
            let mask_val = mask.val();
            let [batch_size, hidden_size] = tensor.dims();

            // For simple sparsity, we take the row sums as a per-neuron mask
            // This approximates the effect of masked weights
            let row_mask: Tensor<B, 1> = mask_val.clone().sum_dim(1).squeeze(1);
            let row_mask_normalized = row_mask.div_scalar(hidden_size as f32);
            let mask_expanded = row_mask_normalized.unsqueeze::<2>().expand([batch_size, hidden_size]);

            tensor.mul(mask_expanded)
        } else {
            tensor
        }
    }

    /// Perform a forward pass through the CfC cell
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        hx: Tensor<B, 2>,
        ts: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = input.dims()[0];
        let device = input.device();

        // Concatenate input and hidden state
        let x = Tensor::cat(vec![input, hx], 1);

        // Compute ff1 and apply sparsity mask
        let ff1_out = self.ff1.forward(x.clone());
        let ff1_out = self.apply_sparsity_mask(ff1_out);

        match self.mode {
            1 => {
                // Pure mode
                let w_tau_layer = self.w_tau.as_ref().unwrap();
                let a_layer = self.a.as_ref().unwrap();

                let ones_input = Tensor::<B, 2>::ones([batch_size, 1], &device);
                let w_tau_out = w_tau_layer.forward(ones_input.clone());
                let a_out = a_layer.forward(ones_input);

                let ts_tensor = Tensor::<B, 2>::full([batch_size, self.hidden_size], ts, &device);
                let abs_w_tau = w_tau_out.abs();
                let abs_ff1 = ff1_out.clone().abs();

                let exp_term = (ts_tensor * (abs_w_tau + abs_ff1)).neg().exp();
                let new_hidden = a_out.clone() - a_out * exp_term * ff1_out;

                (new_hidden.clone(), new_hidden)
            }
            _ => {
                // Default or NoGate mode
                let ff2_out = self.ff2.as_ref().unwrap().forward(x.clone());
                let ff2_out = self.apply_sparsity_mask(ff2_out);

                let ff1_tanh = ff1_out.tanh();
                let ff2_tanh = ff2_out.tanh();

                let time_a = self.time_a.as_ref().unwrap().forward(x.clone());
                let time_b = self.time_b.as_ref().unwrap().forward(x);

                // Compute time interpolation
                let ts_tensor = Tensor::<B, 2>::full([batch_size, self.hidden_size], ts, &device);
                let t_interp = activation::sigmoid(time_a * ts_tensor + time_b);

                let new_hidden = if self.mode == 2 {
                    // NoGate: h = ff1 + t_interp * ff2
                    ff1_tanh + t_interp * ff2_tanh
                } else {
                    // Default: h = ff1 * (1 - t_interp) + t_interp * ff2
                    ff1_tanh
                        * (Tensor::<B, 2>::ones([batch_size, self.hidden_size], &device)
                            - t_interp.clone())
                        + t_interp * ff2_tanh
                };

                (new_hidden.clone(), new_hidden)
            }
        }
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
    fn test_cfc_cell_creation() {
        let device = get_test_device();
        let cell = CfCCell::<TestBackend>::new(20, 50, &device);

        assert_eq!(cell.input_size(), 20);
        assert_eq!(cell.hidden_size(), 50);
        assert_eq!(cell.mode(), CfcMode::Default);
    }

    #[test]
    fn test_cfc_forward_default() {
        let device = get_test_device();
        let cell = CfCCell::<TestBackend>::new(20, 50, &device);

        let batch_size = 4;
        let input = Tensor::<TestBackend, 2>::zeros([batch_size, 20], &device);
        let hx = Tensor::<TestBackend, 2>::zeros([batch_size, 50], &device);

        let (output, new_hidden) = cell.forward(input, hx, 1.0);

        assert_eq!(output.dims(), [batch_size, 50]);
        assert_eq!(new_hidden.dims(), [batch_size, 50]);
    }

    #[test]
    fn test_cfc_forward_pure() {
        let device = get_test_device();
        let cell = CfCCell::<TestBackend>::new(20, 50, &device).with_mode(CfcMode::Pure);

        assert_eq!(cell.mode(), CfcMode::Pure);

        let input = Tensor::<TestBackend, 2>::zeros([2, 20], &device);
        let hx = Tensor::<TestBackend, 2>::zeros([2, 50], &device);

        let (output, _) = cell.forward(input, hx, 1.0);

        assert_eq!(output.dims(), [2, 50]);
    }

    #[test]
    fn test_cfc_forward_no_gate() {
        let device = get_test_device();
        let cell = CfCCell::<TestBackend>::new(20, 50, &device).with_mode(CfcMode::NoGate);

        assert_eq!(cell.mode(), CfcMode::NoGate);

        let input = Tensor::<TestBackend, 2>::ones([2, 20], &device);
        let hx = Tensor::<TestBackend, 2>::zeros([2, 50], &device);

        let (output, new_hidden) = cell.forward(input, hx, 1.0);

        assert_eq!(output.dims(), [2, 50]);
        assert_eq!(new_hidden.dims(), [2, 50]);
    }

    #[test]
    fn test_cfc_state_change() {
        let device = get_test_device();
        let cell = CfCCell::<TestBackend>::new(20, 50, &device);

        let input = Tensor::<TestBackend, 2>::ones([2, 20], &device);
        let hx = Tensor::<TestBackend, 2>::zeros([2, 50], &device);

        let (output, new_hidden) = cell.forward(input, hx.clone(), 1.0);

        // State should have changed
        let diff = (new_hidden.clone() - hx).abs().mean().into_scalar();
        assert!(diff > 0.0, "State should change after forward pass");

        // Output should equal new_hidden for CfC
        let output_diff = (output - new_hidden).abs().mean().into_scalar();
        assert!(output_diff < 1e-6, "Output should equal new_hidden");
    }

    #[test]
    fn test_cfc_different_modes_produce_different_results() {
        let device = get_test_device();

        let cell_default = CfCCell::<TestBackend>::new(20, 50, &device);
        let cell_no_gate = CfCCell::<TestBackend>::new(20, 50, &device).with_mode(CfcMode::NoGate);

        let input = Tensor::<TestBackend, 2>::random(
            [2, 20],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let hx = Tensor::<TestBackend, 2>::zeros([2, 50], &device);

        let (out1, _) = cell_default.forward(input.clone(), hx.clone(), 1.0);
        let (out2, _) = cell_no_gate.forward(input, hx, 1.0);

        let diff = (out1 - out2).abs().mean().into_scalar();
        assert!(
            diff > 0.01,
            "Different modes should produce different outputs"
        );
    }

    #[test]
    fn test_cfc_backbone_configurations() {
        let device = get_test_device();

        // These should not panic (backbone is currently no-op)
        let _cell_no_backbone =
            CfCCell::<TestBackend>::new(20, 50, &device).with_backbone(0, 0, 0.0);

        let _cell_deep_backbone =
            CfCCell::<TestBackend>::new(20, 50, &device).with_backbone(64, 3, 0.2);
    }

    #[test]
    fn test_cfc_activations() {
        let device = get_test_device();

        for activation in ["relu", "tanh", "gelu", "silu", "lecun_tanh"] {
            let cell = CfCCell::<TestBackend>::new(20, 50, &device)
                .with_backbone(64, 1, 0.0)
                .with_activation(activation);

            let input = Tensor::<TestBackend, 2>::zeros([2, 20], &device);
            let hx = Tensor::<TestBackend, 2>::zeros([2, 50], &device);

            let (output, _) = cell.forward(input, hx, 1.0);
            assert_eq!(output.dims()[0], 2);
        }
    }

    #[test]
    #[should_panic]
    fn test_cfc_invalid_activation() {
        let device = get_test_device();
        let _cell =
            CfCCell::<TestBackend>::new(20, 50, &device).with_activation("invalid_activation");
    }

    #[test]
    fn test_cfc_batch_processing() {
        let device = get_test_device();
        let cell = CfCCell::<TestBackend>::new(20, 50, &device);

        // Test with batch sizes 1, 8, 32
        for batch in [1, 8, 32] {
            let input = Tensor::<TestBackend, 2>::zeros([batch, 20], &device);
            let hx = Tensor::<TestBackend, 2>::zeros([batch, 50], &device);

            let (output, _) = cell.forward(input, hx, 1.0);
            assert_eq!(output.dims(), [batch, 50]);
        }
    }

    #[test]
    fn test_cfc_sparsity_mask() {
        let device = get_test_device();
        let mask = Array2::from_shape_vec((50, 50), vec![1.0f32; 2500]).unwrap();

        let cell = CfCCell::<TestBackend>::new(20, 50, &device).with_sparsity_mask(mask, &device);

        assert!(cell.has_sparsity_mask());

        let input = Tensor::<TestBackend, 2>::zeros([2, 20], &device);
        let hx = Tensor::<TestBackend, 2>::zeros([2, 50], &device);

        let (output, _) = cell.forward(input, hx, 1.0);
        assert_eq!(output.dims(), [2, 50]);
    }

    #[test]
    fn test_cfc_from_wiring() {
        let device = get_test_device();
        let wiring = crate::wirings::FullyConnected::new(50, None, 1234, true);

        let cell = CfCCell::<TestBackend>::from_wiring(20, &wiring, &device);

        assert!(cell.has_sparsity_mask());
        assert_eq!(cell.input_size(), 20);
        assert_eq!(cell.hidden_size(), 50);

        let input = Tensor::<TestBackend, 2>::zeros([2, 20], &device);
        let hx = Tensor::<TestBackend, 2>::zeros([2, 50], &device);

        let (output, _) = cell.forward(input, hx, 1.0);
        assert_eq!(output.dims(), [2, 50]);
    }
}
