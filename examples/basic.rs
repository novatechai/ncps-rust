//! Basic usage example of NCPS with CfC network
//!
//! This example demonstrates how to create and use a CfC (Closed-form Continuous-time)
//! recurrent neural network for sequence processing.

use burn::backend::NdArray;
use burn::tensor::Tensor;
use ncps::rnn::CfC;

fn main() {
    println!("=== NCPS Basic Example ===\n");

    // Use the NdArray backend (CPU)
    type Backend = NdArray<f32>;
    let device = Default::default();

    // Example 1: Simple fully-connected CfC (batch-first by default)
    println!("Example 1: Batch-first sequence");
    let cfc = CfC::<Backend>::new(20, 50, &device);

    println!("Created CfC network:");
    println!("  Input size: 20");
    println!("  Hidden size: 50");
    println!();

    // Input shape: [batch=4, seq=10, features=20]
    let input = Tensor::<Backend, 3>::random(
        [4, 10, 20],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let (output, state) = cfc.forward(input, None, None);

    println!("  Input shape:  [4, 10, 20]");
    println!("  Output shape: {:?}", output.dims());
    println!("  State shape:  {:?}", state.dims());
    println!();

    // Example 2: Sequence-first processing
    println!("Example 2: Sequence-first processing");
    let cfc_seq = CfC::<Backend>::new(20, 32, &device).with_batch_first(false);

    // Input shape: [seq=10, batch=2, features=20]
    let input_seq = Tensor::<Backend, 3>::random(
        [10, 2, 20],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let (output_seq, state_seq) = cfc_seq.forward(input_seq, None, None);

    println!("  Input shape:  [10, 2, 20]");
    println!("  Output shape: {:?}", output_seq.dims());
    println!();

    // Example 3: Return only last timestep
    println!("Example 3: Last timestep only");
    let cfc_last = CfC::<Backend>::new(20, 40, &device).with_return_sequences(false);

    let (output_last, _) = cfc_last.forward(
        Tensor::<Backend, 3>::random(
            [4, 10, 20],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        ),
        None,
        None,
    );

    println!("  Input shape:  [4, 10, 20]");
    println!("  Output shape: {:?}", output_last.dims());
    println!("  Only the last timestep is returned");
    println!();

    // Example 4: With wiring configuration (NCP)
    println!("Example 4: Using NCP wiring");
    use ncps::wirings::AutoNCP;
    let wiring = AutoNCP::new(50, 10, 0.5, 12345);
    let cfc_wired = CfC::<Backend>::with_wiring(20, wiring, &device);

    let (output_wired, _) = cfc_wired.forward(
        Tensor::<Backend, 3>::random(
            [2, 5, 20],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        ),
        None,
        None,
    );

    println!("  Input shape:  [2, 5, 20]");
    println!("  Output shape: {:?}", output_wired.dims());
    println!("  Output dimension is motor_size (10)");
    println!();

    println!("=== Examples completed successfully! ===");
}
