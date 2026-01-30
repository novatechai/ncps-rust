//! Custom activation functions for NCPS
//!
//! This module provides activation functions not available in Burn's standard library.

use burn::tensor::{backend::Backend, Tensor};

/// LeCun's tanh activation function.
///
/// This activation function is defined as:
/// `f(x) = 1.7159 * tanh(0.666 * x)`
///
/// It provides a smoother alternative to standard tanh with better gradient flow
/// properties. The scaling factors (1.7159 and 0.666) are chosen such that:
/// - The function approximates the identity near the origin
/// - The output range is approximately [-1.7159, 1.7159]
///
/// # Example
///
/// ```rust
/// use burn::backend::NdArray;
/// use burn::tensor::Tensor;
/// use ncps::activation::LeCun;
///
/// type Backend = NdArray<f32>;
/// let device = Default::default();
///
/// let x = Tensor::<Backend, 1>::from_floats([0.0, 1.0, -1.0], &device);
/// let y = LeCun::forward(x);
/// ```
pub struct LeCun;

impl LeCun {
    /// Applies the LeCun tanh activation function.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of any dimension
    ///
    /// # Returns
    ///
    /// Tensor with LeCun activation applied element-wise
    pub fn forward<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
        // LeCun tanh: 1.7159 * tanh(0.666 * x)
        let scaled = x * 0.666f32;
        scaled.tanh() * 1.7159f32
    }
}

/// Applies LeCun activation to a tensor.
///
/// This is a convenience trait extension for applying LeCun activation directly on tensors.
pub trait LeCunActivation {
    /// Applies LeCun activation
    fn lecun(self) -> Self;
}

impl<B: Backend, const D: usize> LeCunActivation for Tensor<B, D> {
    fn lecun(self) -> Self {
        LeCun::forward(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

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
        // Extract data using slice approach
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
    fn test_lecun_trait() {
        let device = Default::default();
        let x = Tensor::<Backend, 1>::from_floats([0.0f32, 1.0, -1.0], &device);

        // Test using the trait extension
        let y_trait = x.clone().lecun();
        let y_direct = LeCun::forward(x);

        // Compare element by element
        for i in 0..3 {
            let t_val = y_trait.clone().slice([i..i + 1]).into_scalar();
            let d_val = y_direct.clone().slice([i..i + 1]).into_scalar();
            assert!((t_val - d_val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lecun_3d_tensor() {
        let device = Default::default();
        let x = Tensor::<Backend, 3>::random(
            [2, 3, 4],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let y = LeCun::forward(x.clone());

        assert_eq!(y.dims(), [2, 3, 4]);

        // Verify output is within expected range by sampling a few values
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let y_val = y
                        .clone()
                        .slice([i..i + 1, j..j + 1, k..k + 1])
                        .into_scalar();
                    assert!(
                        y_val >= -1.716 && y_val <= 1.716,
                        "Value out of range: {}",
                        y_val
                    );
                }
            }
        }
    }
}
