//! Wired CfC Cell Integration Tests
//!
//! Tests for the WiredCfCCell module following the test specification in TASKS.md

use burn::backend::NdArray;
use burn::tensor::Tensor;
use ncps::cells::{CfcMode, WiredCfCCell};
use ncps::wirings::{AutoNCP, Wiring, NCP};

type Backend = NdArray<f32>;

fn create_wired_cell() -> WiredCfCCell<Backend> {
    let device = Default::default();
    let mut wiring = AutoNCP::new(32, 8, 0.5, 22222);
    wiring.build(16);

    WiredCfCCell::new(&wiring, &device, CfcMode::Default)
}

#[test]
fn test_wired_cfc_creation() {
    let cell = create_wired_cell();

    assert_eq!(cell.state_size(), 32);
    assert_eq!(cell.motor_size(), 8);
    assert_eq!(cell.num_layers(), 3);
}

#[test]
fn test_wired_cfc_layer_sizes() {
    let cell = create_wired_cell();
    let sizes = cell.layer_sizes();

    // Should have 3 layers
    assert_eq!(sizes.len(), 3);
    // Total should equal state_size
    let total: usize = sizes.iter().sum();
    assert_eq!(total, cell.state_size());
}

#[test]
fn test_wired_cfc_forward() {
    let device = Default::default();
    let cell = create_wired_cell();

    let batch_size = 4;
    let input = Tensor::<Backend, 2>::zeros([batch_size, 16], &device);
    let hx = Tensor::<Backend, 2>::zeros([batch_size, 32], &device);

    let (output, new_hidden) = cell.forward(input, hx, 1.0);

    // Output should be motor_size
    assert_eq!(output.dims(), [batch_size, 8]);
    // New hidden should preserve full state
    assert_eq!(new_hidden.dims(), [batch_size, 32]);
}

#[test]
fn test_wired_cfc_state_partitioning() {
    let device = Default::default();
    let cell = create_wired_cell();

    // Create state with different values for each layer
    let layer_sizes = cell.layer_sizes().to_vec();
    let hx_parts: Vec<Tensor<Backend, 2>> = layer_sizes
        .iter()
        .enumerate()
        .map(|(i, &size)| Tensor::<Backend, 2>::full([2, size], (i + 1) as f32, &device))
        .collect();

    let hx = Tensor::cat(hx_parts, 1);

    let input = Tensor::<Backend, 2>::zeros([2, 16], &device);
    let (output, new_hidden) = cell.forward(input, hx, 1.0);

    // Verify state was processed correctly
    assert_eq!(new_hidden.dims(), [2, 32]);
    assert_eq!(output.dims(), [2, 8]);
}

#[test]
fn test_wired_cfc_with_different_wirings() {
    let device = Default::default();

    // Test with manually configured NCP
    let mut wiring = NCP::new(10, 8, 5, 6, 6, 4, 6, 22222);
    wiring.build(10);
    let cell = WiredCfCCell::<Backend>::new(&wiring, &device, CfcMode::Default);

    assert_eq!(cell.state_size(), 23); // 10 + 8 + 5
    assert_eq!(cell.num_layers(), 3);
}

#[test]
fn test_wired_cfc_information_flow() {
    let device = Default::default();
    let cell = create_wired_cell();

    // Test that information flows from sensory through all layers
    let input1 = Tensor::<Backend, 2>::zeros([1, 16], &device);
    let input2 = Tensor::<Backend, 2>::ones([1, 16], &device);
    let hx = Tensor::<Backend, 2>::zeros([1, 32], &device);

    let (out1, _) = cell.forward(input1, hx.clone(), 1.0);
    let (out2, _) = cell.forward(input2, hx, 1.0);

    let diff = (out1 - out2).abs().sum().into_scalar();
    assert!(
        diff > 0.0,
        "Different inputs should produce different outputs"
    );
}
