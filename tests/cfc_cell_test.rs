#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;
    use ncps::cells::cfc_cell::{CfCCell, CfcMode};

    type Backend = NdArray<f32>;

    fn create_test_cell(mode: CfcMode) -> CfCCell<Backend> {
        let device = Default::default();
        CfCCell::new(20, 50, &device)
            .with_mode(mode)
            .with_backbone(128, 1, 0.0)
            .with_activation("lecun_tanh")
    }

    #[test]
    fn test_cfc_cell_creation() {
        let device = Default::default();
        let cell = CfCCell::<Backend>::new(20, 50, &device);

        assert_eq!(cell.input_size(), 20);
        assert_eq!(cell.hidden_size(), 50);
        assert_eq!(cell.mode(), CfcMode::Default);
    }

    #[test]
    fn test_cfc_forward_default() {
        let device = Default::default();
        let cell = create_test_cell(CfcMode::Default);

        let batch_size = 4;
        let input = Tensor::<Backend, 2>::zeros([batch_size, 20], &device);
        let hx = Tensor::<Backend, 2>::zeros([batch_size, 50], &device);

        let (output, new_hidden) = cell.forward(input, hx, 1.0);

        // Output is hidden_size, state is hidden_size
        assert_eq!(output.dims(), [batch_size, 50]);
        assert_eq!(new_hidden.dims(), [batch_size, 50]);
    }

    #[test]
    fn test_cfc_forward_pure() {
        let device = Default::default();
        let cell = create_test_cell(CfcMode::Pure);

        let input = Tensor::<Backend, 2>::random(
            [2, 20],
            burn::tensor::Distribution::Uniform(-0.5, 0.5),
            &device,
        );
        let hx = Tensor::<Backend, 2>::zeros([2, 50], &device);

        let (output, _) = cell.forward(input, hx, 1.0);

        // Pure mode should return correct shape output
        assert_eq!(output.dims(), [2, 50]);
        assert_eq!(cell.mode(), CfcMode::Pure);
    }

    #[test]
    fn test_cfc_forward_no_gate() {
        let device = Default::default();
        let cell = create_test_cell(CfcMode::NoGate);

        let input = Tensor::<Backend, 2>::ones([2, 20], &device);
        let hx = Tensor::<Backend, 2>::zeros([2, 50], &device);

        let (output, new_hidden) = cell.forward(input, hx, 1.0);

        // NoGate mode should return correct shape output
        assert_eq!(output.dims(), [2, 50]);
        assert_eq!(new_hidden.dims(), [2, 50]);
        assert_eq!(cell.mode(), CfcMode::NoGate);
    }

    #[test]
    fn test_cfc_state_change() {
        let device = Default::default();
        let cell = create_test_cell(CfcMode::Default);

        let input = Tensor::<Backend, 2>::ones([2, 20], &device);
        let hx = Tensor::<Backend, 2>::zeros([2, 50], &device);

        let (output, new_hidden) = cell.forward(input, hx.clone(), 1.0);

        // State should have changed
        let diff = (new_hidden.clone() - hx).abs().mean().into_scalar();
        assert!(diff > 0.0);

        // Output should equal new_hidden for CfC
        let output_diff = (output - new_hidden).abs().mean().into_scalar();
        assert!(output_diff < 1e-6, "Output should equal new_hidden");
    }

    #[test]
    fn test_cfc_different_modes_produce_different_results() {
        let device = Default::default();

        let cell_default = create_test_cell(CfcMode::Default);
        let cell_no_gate = create_test_cell(CfcMode::NoGate);

        let input = Tensor::<Backend, 2>::random(
            [2, 20],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let hx = Tensor::<Backend, 2>::zeros([2, 50], &device);

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
        let device = Default::default();

        // Test different backbone configurations - just ensure they're created without panic
        let _cell_no_backbone = CfCCell::<Backend>::new(20, 50, &device).with_backbone(0, 0, 0.0); // No backbone

        let _cell_deep_backbone =
            CfCCell::<Backend>::new(20, 50, &device).with_backbone(64, 3, 0.2); // Deep with dropout

        // Test forward passes work
        let input = Tensor::<Backend, 2>::zeros([2, 20], &device);
        let hx = Tensor::<Backend, 2>::zeros([2, 50], &device);

        let _cell = CfCCell::<Backend>::new(20, 50, &device).with_backbone(0, 0, 0.0);
        let (out1, _) = _cell.forward(input.clone(), hx.clone(), 1.0);
        assert_eq!(out1.dims(), [2, 50]);

        let _cell2 = CfCCell::<Backend>::new(20, 50, &device).with_backbone(64, 3, 0.2);
        let (out2, _) = _cell2.forward(input, hx, 1.0);
        assert_eq!(out2.dims(), [2, 50]);
    }

    #[test]
    fn test_cfc_activations() {
        let device = Default::default();

        for activation in ["relu", "tanh", "gelu", "silu", "lecun_tanh"] {
            let cell = CfCCell::<Backend>::new(20, 50, &device)
                .with_backbone(64, 1, 0.0)
                .with_activation(activation);

            let input = Tensor::<Backend, 2>::zeros([2, 20], &device);
            let hx = Tensor::<Backend, 2>::zeros([2, 50], &device);

            let (output, _) = cell.forward(input, hx, 1.0);
            assert_eq!(output.dims()[0], 2);
        }
    }

    #[test]
    #[should_panic]
    fn test_cfc_invalid_activation() {
        let device = Default::default();
        CfCCell::<Backend>::new(20, 50, &device).with_activation("invalid_activation");
    }

    #[test]
    fn test_cfc_batch_processing() {
        let device = Default::default();
        let cell = create_test_cell(CfcMode::Default);

        // Test with batch sizes 1, 8, 32
        for batch in [1, 8, 32] {
            let input = Tensor::<Backend, 2>::zeros([batch, 20], &device);
            let hx = Tensor::<Backend, 2>::zeros([batch, 50], &device);

            let (output, _) = cell.forward(input, hx, 1.0);
            assert_eq!(output.dims(), [batch, 50]);
        }
    }

    #[test]
    fn test_cfc_sparsity_mask() {
        let device = Default::default();
        use ndarray::Array2;

        let mask = Array2::from_shape_vec((50, 50), vec![1.0f32; 2500]).unwrap();

        let cell = CfCCell::<Backend>::new(20, 50, &device).with_sparsity_mask(mask, &device);

        // Ensure forward pass works with mask configured
        let input = Tensor::<Backend, 2>::zeros([2, 20], &device);
        let hx = Tensor::<Backend, 2>::zeros([2, 50], &device);

        let (output, _) = cell.forward(input, hx, 1.0);
        assert_eq!(output.dims(), [2, 50]);
    }
}
