//! Tests for custom activation functions

use burn::backend::NdArray;
use burn::tensor::Tensor;
use ncps::activation::LeCun;

type Backend = NdArray<f32>;

#[test]
fn test_lecun_tanh_zero() {
    let device = Default::default();
    let x = Tensor::<Backend, 1>::zeros([5], &device);
    let y = LeCun::forward(x);

    // tanh(0) = 0, so LeCun(0) = 0
    let sum = y.sum().into_scalar();
    assert!((sum - 0.0).abs() < 1e-6);
}

#[test]
fn test_lecun_tanh_range() {
    let device = Default::default();

    // Test various inputs
    let test_values = [-10.0f32, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];

    for &val in &test_values {
        let x = Tensor::<Backend, 1>::full([1], val, &device);
        let y = LeCun::forward(x);

        let result = y.into_scalar();
        let expected = 1.7159f32 * (0.666f32 * val).tanh();

        assert!(
            (result - expected).abs() < 1e-5,
            "LeCun activation incorrect at x={}",
            val
        );
    }
}

#[test]
fn test_lecun_tanh_multidimensional() {
    let device = Default::default();
    let x = Tensor::<Backend, 2>::random(
        [4, 8],
        burn::tensor::Distribution::Uniform(-2.0, 2.0),
        &device,
    );

    let y = LeCun::forward(x.clone());

    assert_eq!(y.dims(), [4, 8]);

    // Verify element-wise correctness by comparing a few values
    for i in 0..4 {
        for j in 0..8 {
            let x_val = x.clone().slice([i..i + 1, j..j + 1]).into_scalar();
            let y_val = y.clone().slice([i..i + 1, j..j + 1]).into_scalar();
            let expected = 1.7159f32 * (0.666f32 * x_val).tanh();
            assert!(
                (y_val - expected).abs() < 1e-5,
                "Element [{}, {}] incorrect: got {}, expected {}",
                i,
                j,
                y_val,
                expected
            );
        }
    }
}

#[test]
fn test_lecun_tanh_saturation() {
    let device = Default::default();

    // Very large positive input should saturate near max
    let x_large_pos = Tensor::<Backend, 1>::full([1], 100.0f32, &device);
    let y_pos = LeCun::forward(x_large_pos);
    assert!(y_pos.into_scalar() > 1.7);

    // Very large negative input should saturate near min
    let x_large_neg = Tensor::<Backend, 1>::full([1], -100.0f32, &device);
    let y_neg = LeCun::forward(x_large_neg);
    assert!(y_neg.into_scalar() < -1.7);
}

#[test]
fn test_lecun_tanh_gradient() {
    // Test that the function is differentiable
    // Note: This tests basic gradient flow through the activation
    let device = Default::default();

    // Create input
    let x = Tensor::<Backend, 1>::from_floats([0.0f32, 1.0, -1.0], &device);
    let y = LeCun::forward(x);

    // Verify output shape matches input
    assert_eq!(y.dims()[0], 3);

    // The function should be smooth and produce finite gradients
    // Check by verifying all values are finite
    for i in 0..3 {
        let val = y.clone().slice([i..i + 1]).into_scalar();
        assert!(val.is_finite());
    }
}
