//! Liquid Time-Constant (LTC) Cell Implementation
//!
//! Reference: Hasani et al., "Liquid time-constant networks", AAAI 2021

use crate::wirings::Wiring;
use burn::module::{Module, Param};
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

/// Input/output mapping modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MappingMode {
    /// Affine mapping: y = w * x + b
    #[default]
    Affine,
    /// Linear mapping: y = w * x
    Linear,
    /// No mapping (pass-through)
    None,
}

/// Liquid Time-Constant (LTC) Cell
#[derive(Debug, Module)]
pub struct LTCCell<B: Backend> {
    /// Leak conductance (must be positive)
    pub gleak: Param<Tensor<B, 1>>,
    /// Leak reversal potential
    pub vleak: Param<Tensor<B, 1>>,
    /// Membrane capacitance (must be positive)
    pub cm: Param<Tensor<B, 1>>,
    /// Sigmoid center parameter for internal synapses
    pub sigma: Param<Tensor<B, 2>>,
    /// Sigmoid steepness parameter for internal synapses
    pub mu: Param<Tensor<B, 2>>,
    /// Synaptic weights for internal synapses (must be positive)
    pub w: Param<Tensor<B, 2>>,
    /// Reversal potentials for internal synapses (from wiring)
    pub erev: Param<Tensor<B, 2>>,
    /// Sigmoid center parameter for sensory synapses
    pub sensory_sigma: Param<Tensor<B, 2>>,
    /// Sigmoid steepness parameter for sensory synapses
    pub sensory_mu: Param<Tensor<B, 2>>,
    /// Synaptic weights for sensory synapses (must be positive)
    pub sensory_w: Param<Tensor<B, 2>>,
    /// Reversal potentials for sensory synapses (from wiring)
    pub sensory_erev: Param<Tensor<B, 2>>,
    /// Sparsity mask for internal synapses (non-trainable)
    pub sparsity_mask: Param<Tensor<B, 2>>,
    /// Sparsity mask for sensory synapses (non-trainable)
    pub sensory_sparsity_mask: Param<Tensor<B, 2>>,
    /// Input weight for mapping
    pub input_w: Option<Param<Tensor<B, 1>>>,
    /// Input bias for mapping
    pub input_b: Option<Param<Tensor<B, 1>>>,
    /// Output weight for mapping
    pub output_w: Option<Param<Tensor<B, 1>>>,
    /// Output bias for mapping
    pub output_b: Option<Param<Tensor<B, 1>>>,
    /// Number of ODE solver steps per forward pass
    #[module(skip)]
    ode_unfolds: usize,
    /// Epsilon for numerical stability
    #[module(skip)]
    epsilon: f64,
    /// State size (number of neurons)
    #[module(skip)]
    state_size: usize,
    /// Motor size (output neurons)
    #[module(skip)]
    motor_size: usize,
    /// Sensory size (input neurons)
    #[module(skip)]
    sensory_size: usize,
    /// Input mapping mode (0=None, 1=Linear, 2=Affine)
    #[module(skip)]
    input_mapping: u8,
    /// Output mapping mode (0=None, 1=Linear, 2=Affine)
    #[module(skip)]
    output_mapping: u8,
}

impl<B: Backend> LTCCell<B> {
    /// Creates a new LTC Cell with the given wiring configuration
    pub fn new(wiring: &dyn Wiring, sensory_size: Option<usize>, device: &B::Device) -> Self {
        let state_size = wiring.units();
        let motor_size = wiring.output_dim().unwrap_or(state_size);
        let actual_sensory_size = sensory_size.or_else(|| wiring.input_dim()).expect(
            "LTCCell requires sensory_size or wiring with input_dim. Call wiring.build() first.",
        );

        // Initialize parameters with specified ranges
        let gleak = Self::init_param([state_size], 0.001, 1.0, device);
        let vleak = Self::init_param([state_size], -0.2, 0.2, device);
        let cm = Self::init_param([state_size], 0.4, 0.6, device);

        // 2D parameters
        let sigma = Self::init_param([state_size, state_size], 3.0, 8.0, device);
        let mu = Self::init_param([state_size, state_size], 0.3, 0.8, device);
        let w = Self::init_param([state_size, state_size], 0.001, 1.0, device);

        // Get erev from wiring adjacency matrix (this encodes excitatory/inhibitory polarity)
        let erev_matrix = wiring.erev_initializer();
        let erev = Self::tensor_from_ndarray(&erev_matrix, device);

        // Get sparsity mask from adjacency matrix (absolute values)
        let sparsity_mask = Self::sparsity_mask_from_ndarray(&erev_matrix, device);

        let sensory_sigma = Self::init_param([actual_sensory_size, state_size], 3.0, 8.0, device);
        let sensory_mu = Self::init_param([actual_sensory_size, state_size], 0.3, 0.8, device);
        let sensory_w = Self::init_param([actual_sensory_size, state_size], 0.001, 1.0, device);

        // Get sensory erev and sparsity mask from wiring
        let (sensory_erev, sensory_sparsity_mask) =
            if let Some(sensory_matrix) = wiring.sensory_erev_initializer() {
                (
                    Self::tensor_from_ndarray(&sensory_matrix, device),
                    Self::sparsity_mask_from_ndarray(&sensory_matrix, device),
                )
            } else {
                // If no sensory adjacency, create fully connected
                (
                    Param::from_tensor(Tensor::ones([actual_sensory_size, state_size], device)),
                    Param::from_tensor(Tensor::ones([actual_sensory_size, state_size], device)),
                )
            };

        Self {
            gleak,
            vleak,
            cm,
            sigma,
            mu,
            w,
            erev,
            sensory_sigma,
            sensory_mu,
            sensory_w,
            sensory_erev,
            sparsity_mask,
            sensory_sparsity_mask,
            input_w: None,
            input_b: None,
            output_w: None,
            output_b: None,
            ode_unfolds: 6,
            epsilon: 1e-8,
            state_size,
            motor_size,
            sensory_size: actual_sensory_size,
            input_mapping: 0,  // MappingMode::None
            output_mapping: 0, // MappingMode::None
        }
    }

    /// Convert ndarray to Burn tensor parameter
    fn tensor_from_ndarray(
        arr: &ndarray::Array2<i32>,
        device: &B::Device,
    ) -> Param<Tensor<B, 2>> {
        let shape = arr.shape();
        let data: Vec<f32> = arr.iter().map(|&x| x as f32).collect();
        let tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([shape[0], shape[1]]);
        Param::from_tensor(tensor)
    }

    /// Create sparsity mask from adjacency matrix (|adjacency|)
    fn sparsity_mask_from_ndarray(
        arr: &ndarray::Array2<i32>,
        device: &B::Device,
    ) -> Param<Tensor<B, 2>> {
        let shape = arr.shape();
        let data: Vec<f32> = arr.iter().map(|&x| x.abs() as f32).collect();
        let tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([shape[0], shape[1]]);
        Param::from_tensor(tensor)
    }

    fn init_param<const D: usize>(
        shape: [usize; D],
        min: f64,
        max: f64,
        device: &B::Device,
    ) -> Param<Tensor<B, D>> {
        let tensor = Tensor::random(shape, Distribution::Uniform(min, max), device);
        Param::from_tensor(tensor)
    }

    pub fn with_ode_unfolds(mut self, unfolds: usize) -> Self {
        self.ode_unfolds = unfolds;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set input mapping mode (affine, linear, or none)
    pub fn with_input_mapping(mut self, mode: MappingMode, device: &B::Device) -> Self {
        self.input_mapping = match mode {
            MappingMode::None => 0,
            MappingMode::Linear => 1,
            MappingMode::Affine => 2,
        };
        match mode {
            MappingMode::Affine => {
                self.input_w =
                    Some(Param::from_tensor(Tensor::ones([self.sensory_size], device)));
                self.input_b =
                    Some(Param::from_tensor(Tensor::zeros([self.sensory_size], device)));
            }
            MappingMode::Linear => {
                self.input_w =
                    Some(Param::from_tensor(Tensor::ones([self.sensory_size], device)));
                self.input_b = None;
            }
            MappingMode::None => {
                self.input_w = None;
                self.input_b = None;
            }
        }
        self
    }

    /// Set output mapping mode (affine, linear, or none)
    pub fn with_output_mapping(mut self, mode: MappingMode, device: &B::Device) -> Self {
        self.output_mapping = match mode {
            MappingMode::None => 0,
            MappingMode::Linear => 1,
            MappingMode::Affine => 2,
        };
        match mode {
            MappingMode::Affine => {
                self.output_w = Some(Param::from_tensor(Tensor::ones([self.motor_size], device)));
                self.output_b = Some(Param::from_tensor(Tensor::zeros([self.motor_size], device)));
            }
            MappingMode::Linear => {
                self.output_w = Some(Param::from_tensor(Tensor::ones([self.motor_size], device)));
                self.output_b = None;
            }
            MappingMode::None => {
                self.output_w = None;
                self.output_b = None;
            }
        }
        self
    }

    pub fn state_size(&self) -> usize {
        self.state_size
    }

    pub fn motor_size(&self) -> usize {
        self.motor_size
    }

    pub fn sensory_size(&self) -> usize {
        self.sensory_size
    }

    pub fn synapse_count(&self) -> usize {
        self.state_size * self.state_size
    }

    pub fn sensory_synapse_count(&self) -> usize {
        self.sensory_size * self.state_size
    }

    /// Apply input mapping
    fn map_inputs(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut result = inputs;
        if let Some(ref w) = self.input_w {
            result = result.mul(w.val().unsqueeze::<2>());
        }
        if let Some(ref b) = self.input_b {
            result = result.add(b.val().unsqueeze::<2>());
        }
        result
    }

    /// Apply output mapping
    fn map_outputs(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        // First slice to motor size
        let mut output = state.narrow(1, 0, self.motor_size);

        if let Some(ref w) = self.output_w {
            output = output.mul(w.val().unsqueeze::<2>());
        }
        if let Some(ref b) = self.output_b {
            output = output.add(b.val().unsqueeze::<2>());
        }
        output
    }

    /// Apply weight constraints (clamp positive parameters to be >= 0)
    pub fn apply_weight_constraints(&mut self) {
        // In implicit mode (default), constraints are applied via softplus
        // This method is for explicit mode where we clamp negative values
        self.w = Param::from_tensor(self.w.val().clamp_min(0.0));
        self.sensory_w = Param::from_tensor(self.sensory_w.val().clamp_min(0.0));
        self.cm = Param::from_tensor(self.cm.val().clamp_min(0.0));
        self.gleak = Param::from_tensor(self.gleak.val().clamp_min(0.0));
    }
}

impl<B: Backend> LTCCell<B> {
    fn softplus_1d(&self, x: Tensor<B, 1>) -> Tensor<B, 1> {
        x.exp().add_scalar(1.0).log()
    }

    fn softplus_2d(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        x.clone().exp().add_scalar(1.0).log()
    }

    fn _ode_solver(
        &self,
        inputs: Tensor<B, 2>,
        state: Tensor<B, 2>,
        elapsed_time: Tensor<B, 1>,
    ) -> Tensor<B, 2> {
        let [batch, state_size] = state.dims();
        let sensory_size = self.sensory_size;
        let mut v_pre = state;

        // Compute cm_t: cm is [state_size], time is [batch]
        // Formula: cm_t = softplus(cm) / (elapsed_time / ode_unfolds)
        let cm = self.softplus_1d(self.cm.val()); // [state_size]

        // Expand cm: [state_size] -> unsqueeze to [1, state_size] -> expand to [batch, state_size]
        let cm_expanded = cm
            .unsqueeze::<2>() // [1, state_size]
            .expand([batch, state_size]); // [batch, state_size]

        // Compute dt per unfold: [batch] -> unsqueeze_dim(1) -> [batch, 1] -> expand to [batch, state_size]
        let dt = elapsed_time.div_scalar(self.ode_unfolds as f64); // [batch]
        let dt_expanded = dt
            .unsqueeze_dim::<2>(1) // [batch, 1]
            .expand([batch, state_size]); // [batch, state_size]

        let cm_t = cm_expanded.div(dt_expanded);

        // Compute sensory activations
        // sensory_sigmoid: [batch, sensory_size, state_size]
        let sensory_sigmoid = self.compute_sensory_sigmoid(&inputs);

        // w * sigmoid(inputs): [batch, sensory_size, state_size]
        let sensory_w_pos = self.softplus_2d(&self.sensory_w.val());
        let sensory_w_expanded = sensory_w_pos.unsqueeze::<3>();
        let sensory_w_activation = sensory_w_expanded.mul(sensory_sigmoid);

        // Apply sensory sparsity mask: [sensory_size, state_size] -> [1, sensory_size, state_size]
        let sensory_mask_expanded = self
            .sensory_sparsity_mask
            .val()
            .reshape([1, sensory_size, state_size]);
        let sensory_w_activation = sensory_w_activation.mul(sensory_mask_expanded);

        // erev * w_activation
        let sensory_erev_expanded = self.sensory_erev.val().unsqueeze::<3>();
        let sensory_rev_activation = sensory_w_activation.clone().mul(sensory_erev_expanded);

        // Sum over sensory dimension
        let w_numerator_sensory: Tensor<B, 2> = sensory_rev_activation.sum_dim(1).squeeze(1);
        let w_denominator_sensory: Tensor<B, 2> = sensory_w_activation.sum_dim(1).squeeze(1);

        let w_pos = self.softplus_2d(&self.w.val());

        // Get sparsity mask for internal synapses: [state_size, state_size] -> [1, state_size, state_size]
        let sparsity_mask_expanded = self
            .sparsity_mask
            .val()
            .reshape([1, state_size, state_size]);

        // ODE iterations
        for _ in 0..self.ode_unfolds {
            // Compute internal synapse activations
            let sigmoid_val = self.compute_sigmoid_2d(&v_pre, &self.mu.val(), &self.sigma.val());

            // w_activation = w_pos * sigmoid_val
            let w_expanded = w_pos.clone().unsqueeze::<3>();
            let w_activation = w_expanded.mul(sigmoid_val);

            // Apply sparsity mask to enforce wiring connectivity
            let w_activation = w_activation.mul(sparsity_mask_expanded.clone());

            // rev_activation = w_activation * erev
            let erev_expanded = self.erev.val().unsqueeze::<3>();
            let rev_activation = w_activation.clone().mul(erev_expanded);

            // Sum over source dimension
            let w_numerator: Tensor<B, 2> = rev_activation
                .sum_dim(1)
                .squeeze(1)
                .add(w_numerator_sensory.clone());
            let w_denominator: Tensor<B, 2> = w_activation
                .sum_dim(1)
                .squeeze(1)
                .add(w_denominator_sensory.clone());

            // Update voltage
            let gleak_pos = self
                .softplus_1d(self.gleak.val())
                .unsqueeze::<2>()
                .expand([batch, state_size]);
            let vleak_expanded = self
                .vleak
                .val()
                .unsqueeze::<2>()
                .expand([batch, state_size]);

            let numerator = cm_t
                .clone()
                .mul(v_pre.clone())
                .add(gleak_pos.clone().mul(vleak_expanded))
                .add(w_numerator);
            let denominator = cm_t
                .clone()
                .add(gleak_pos)
                .add(w_denominator)
                .add_scalar(self.epsilon);

            v_pre = numerator.div(denominator);
        }

        v_pre
    }

    fn compute_sigmoid_2d(
        &self,
        v_pre: &Tensor<B, 2>,
        mu: &Tensor<B, 2>,
        sigma: &Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [batch, state_size] = v_pre.dims();

        // v_pre: [batch, state_size] -> [batch, state_size, 1]
        // mu, sigma: [state_size, state_size]
        let v_expanded = v_pre.clone().reshape([batch, state_size, 1]);
        let mu_expanded = mu.clone().reshape([1, state_size, state_size]);
        let sigma_expanded = sigma.clone().reshape([1, state_size, state_size]);

        let diff = v_expanded.sub(mu_expanded);
        let scaled = sigma_expanded.mul(diff);

        activation::sigmoid(scaled.reshape([batch * state_size, state_size]))
            .reshape([batch, state_size, state_size])
    }

    fn compute_sensory_sigmoid(&self, inputs: &Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, sensory_size] = inputs.dims();
        let state_size = self.state_size;

        // inputs: [batch, sensory_size] -> [batch, sensory_size, 1]
        let inputs_expanded = inputs.clone().reshape([batch, sensory_size, 1]);
        let mu_expanded = self.sensory_mu.val().reshape([1, sensory_size, state_size]);
        let sigma_expanded = self
            .sensory_sigma
            .val()
            .reshape([1, sensory_size, state_size]);

        let diff = inputs_expanded.sub(mu_expanded);
        let scaled = sigma_expanded.mul(diff);

        activation::sigmoid(scaled.reshape([batch * sensory_size, state_size])).reshape([
            batch,
            sensory_size,
            state_size,
        ])
    }

    pub fn forward(
        &self,
        inputs: Tensor<B, 2>,
        states: Tensor<B, 2>,
        elapsed_time: Tensor<B, 1>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Apply input mapping
        let mapped_inputs = self.map_inputs(inputs);

        // Run ODE solver
        let new_states = self._ode_solver(mapped_inputs, states, elapsed_time);

        // Apply output mapping
        let output = self.map_outputs(new_states.clone());

        (output, new_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type Backend = NdArray<f32>;

    fn create_test_cell() -> LTCCell<Backend> {
        let device = Default::default();
        let wiring = crate::wirings::FullyConnected::new(10, Some(5), 1234, true);

        LTCCell::new(&wiring, Some(8), &device)
            .with_ode_unfolds(6)
            .with_epsilon(1e-8)
    }

    #[test]
    fn test_ltc_cell_creation() {
        let device = Default::default();
        let wiring = crate::wirings::FullyConnected::new(10, Some(5), 1234, true);
        let cell = LTCCell::<Backend>::new(&wiring, Some(8), &device);

        assert_eq!(cell.state_size(), 10);
        assert_eq!(cell.motor_size(), 5);
        assert_eq!(cell.sensory_size(), 8);
    }

    #[test]
    fn test_ltc_cell_forward() {
        let device = Default::default();
        let cell = create_test_cell();

        let batch_size = 4;
        let inputs = Tensor::<Backend, 2>::zeros([batch_size, 8], &device);
        let states = Tensor::<Backend, 2>::zeros([batch_size, 10], &device);
        let elapsed_time = Tensor::<Backend, 1>::ones([batch_size], &device);

        let (output, new_state) = cell.forward(inputs, states, elapsed_time);

        assert_eq!(output.dims(), [batch_size, 5]);
        assert_eq!(new_state.dims(), [batch_size, 10]);
    }

    #[test]
    fn test_ltc_state_change() {
        let device = Default::default();
        let cell = create_test_cell();

        let inputs =
            Tensor::<Backend, 2>::random([2, 8], Distribution::Uniform(-1.0, 1.0), &device);
        let states = Tensor::<Backend, 2>::zeros([2, 10], &device);
        let elapsed_time = Tensor::<Backend, 1>::full([2], 1.0, &device);

        let (output, new_state) =
            cell.forward(inputs.clone(), states.clone(), elapsed_time.clone());

        // State should have changed
        let state_diff = new_state.abs().mean().into_scalar();
        assert!(state_diff > 0.0, "State should change after forward pass");
    }
}
