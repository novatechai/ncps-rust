//! Integration tests for the LTC Cell

use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor};
use ncps::cells::LTCCell;
use ncps::wirings::{AutoNCP, FullyConnected};

type Backend = NdArray<f32>;

fn create_test_cell() -> LTCCell<Backend> {
    let device = Default::default();
    let wiring = FullyConnected::new(10, Some(5), 1234, true);

    LTCCell::new(&wiring, Some(8), &device)
        .with_ode_unfolds(6)
        .with_epsilon(1e-8)
}

#[test]
fn test_ltc_cell_creation() {
    let device = Default::default();
    let wiring = FullyConnected::new(10, Some(5), 1234, true);
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

    // Output is motor_size, state is state_size
    assert_eq!(output.dims(), [batch_size, 5]);
    assert_eq!(new_state.dims(), [batch_size, 10]);
}

#[test]
fn test_ltc_state_change() {
    let device = Default::default();
    let cell = create_test_cell();

    let inputs = Tensor::<Backend, 2>::random([2, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let states = Tensor::<Backend, 2>::zeros([2, 10], &device);
    let elapsed_time = Tensor::<Backend, 1>::full([2], 1.0, &device);

    let (output, new_state) = cell.forward(inputs.clone(), states.clone(), elapsed_time.clone());

    // State should have changed
    let state_diff = new_state.abs().mean().into_scalar();
    assert!(state_diff > 0.0, "State should change after forward pass");
}

#[test]
fn test_ltc_synapse_count() {
    let device = Default::default();
    let wiring = FullyConnected::new(10, Some(5), 1234, true);
    let cell = LTCCell::<Backend>::new(&wiring, Some(8), &device);

    // Synapse counts should be as expected
    assert_eq!(cell.synapse_count(), 100); // state_size^2
    assert_eq!(cell.sensory_synapse_count(), 80); // sensory_size * state_size
}

#[test]
fn test_ltc_parameter_shapes() {
    let device = Default::default();
    let wiring = FullyConnected::new(5, Some(3), 1234, true);
    let cell = LTCCell::<Backend>::new(&wiring, Some(4), &device);

    // Check dimensions
    assert_eq!(cell.state_size(), 5);
    assert_eq!(cell.motor_size(), 3);
    assert_eq!(cell.sensory_size(), 4);
}

#[test]
fn test_ltc_different_time_steps() {
    let device = Default::default();
    let cell = create_test_cell();

    let inputs = Tensor::<Backend, 2>::ones([2, 8], &device);
    let states = Tensor::<Backend, 2>::zeros([2, 10], &device);

    // Small time step
    let ts_small = Tensor::<Backend, 1>::full([2], 0.1, &device);
    let (out1, state1) = cell.forward(inputs.clone(), states.clone(), ts_small);

    // Large time step
    let ts_large = Tensor::<Backend, 1>::full([2], 1.0, &device);
    let (out2, state2) = cell.forward(inputs, states, ts_large);

    // Different time steps should produce different results
    let diff = (state1 - state2).abs().mean().into_scalar();
    assert!(
        diff > 0.01,
        "Different time steps should produce different states"
    );
}

#[test]
fn test_ltc_with_ncp_wiring() {
    use ncps::wirings::AutoNCP;

    let device = Default::default();
    let wiring = AutoNCP::new(32, 8, 0.5, 22222);
    let cell = LTCCell::<Backend>::new(&wiring, Some(16), &device);

    assert_eq!(cell.state_size(), 32);
    assert_eq!(cell.motor_size(), 8);

    let inputs = Tensor::<Backend, 2>::zeros([2, 16], &device);
    let states = Tensor::<Backend, 2>::zeros([2, 32], &device);
    let elapsed_time = Tensor::<Backend, 1>::ones([2], &device);

    let (output, new_state) = cell.forward(inputs, states, elapsed_time);

    assert_eq!(output.dims(), [2, 8]);
    assert_eq!(new_state.dims(), [2, 32]);
}

#[test]
fn test_ltc_different_batch_sizes() {
    let device = Default::default();
    let wiring = FullyConnected::new(10, Some(5), 1234, true);
    let cell = LTCCell::<Backend>::new(&wiring, Some(8), &device);

    for batch_size in [1, 4, 16, 32] {
        let inputs = Tensor::<Backend, 2>::zeros([batch_size, 8], &device);
        let states = Tensor::<Backend, 2>::zeros([batch_size, 10], &device);
        let elapsed_time = Tensor::<Backend, 1>::ones([batch_size], &device);

        let (output, new_state) = cell.forward(inputs, states, elapsed_time);

        assert_eq!(output.dims(), [batch_size, 5]);
        assert_eq!(new_state.dims(), [batch_size, 10]);
    }
}

#[test]
fn test_ltc_with_random_inputs() {
    let device = Default::default();
    let wiring = FullyConnected::new(10, Some(5), 1234, true);
    let cell = LTCCell::<Backend>::new(&wiring, Some(8), &device);

    let inputs = Tensor::<Backend, 2>::random([4, 8], Distribution::Uniform(-2.0, 2.0), &device);
    let states = Tensor::<Backend, 2>::zeros([4, 10], &device);
    let elapsed_time = Tensor::<Backend, 1>::ones([4], &device);

    let (output, new_state) = cell.forward(inputs, states, elapsed_time);

    assert_eq!(output.dims(), [4, 5]);
    assert_eq!(new_state.dims(), [4, 10]);

    // Output should be finite
    let output_sum = output.sum().into_scalar();
    assert!(output_sum.is_finite(), "Output should be finite");
}
