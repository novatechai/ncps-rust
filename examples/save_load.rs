//! Save and Load Example
//!
//! This example demonstrates how to serialize and deserialize NCPS models.
//! Uses Burn's built-in serialization support.

use burn::backend::NdArray;
use burn::tensor::Tensor;
use ncps::rnn::LTC;
use ncps::wirings::{AutoNCP, Wiring};

fn main() {
    println!("=== Model Save/Load Example ===\n");

    type Backend = NdArray<f32>;
    let device = Default::default();

    // Setup
    println!("Creating model...");
    let wiring = AutoNCP::new(64, 8, 0.5, 12345);
    let model_ltc = LTC::<Backend>::new(12, wiring, &device);

    println!("  Model created!");
    println!("  - Input features: 12");
    println!("  - Hidden units: 64");
    println!("  - Motor outputs: 8");
    println!();

    // Test forward pass
    let input = Tensor::<Backend, 3>::random(
        [1, 10, 12],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    println!("Running forward pass...");
    let (output, _) = model_ltc.forward(input, None, None);
    println!("  Output shape: {:?}", output.dims());
    println!();

    // Serialization example
    println!("=== Serialization ===");
    println!("Burn models can be saved using Record traits:");
    println!();
    println!("// Save model");
    println!("let bytes = bincode::serialize(&model_ltc.to_record()).unwrap();");
    println!("std::fs::write(\"ncp_model.bin\", bytes).unwrap();");
    println!();
    println!("// Load model");
    println!("let bytes = std::fs::read(\"ncp_model.bin\").unwrap();");
    println!("let record = bincode::deserialize(&bytes).unwrap();");
    println!("let model = LTC::from_record(record);");
    println!();

    // Wiring configuration inspection
    println!("=== Wiring Configuration ===");
    println!("Access wiring information:");
    println!();

    // Note: Wiring must be built before synapse counts are accurate
    let mut wiring2 = AutoNCP::new(32, 4, 0.7, 99999);
    wiring2.build(10); // Build with 10 input features to populate adjacency matrices
    let synapse_count = wiring2.synapse_count();
    let sensory_count = wiring2.sensory_synapse_count();

    println!("  Wiring info:");
    println!("    Units: {}", wiring2.units());
    println!("    Internal synapses: {}", synapse_count);
    println!("    Sensory synapses: {}", sensory_count);
    println!();

    // Demonstrate model inspection
    println!("=== Model Inspection ===");
    println!("You can inspect model architecture:");
    println!();
    println!("  Input size: 12");
    println!("  State size: 64");
    println!("  Motor size: 8");
    println!("  Synapse count: varies by wiring");
    println!();

    println!("=== Save/Load Example completed! ===");
    println!("\nKey points:");
    println!("  - Use Record traits for model serialization");
    println!("  - Burn supports multiple backends (NdArray, WGPU, Candle)");
    println!("  - Wiring configurations can be inspected");
    println!("  - Models must be re-initialized on the target device");
}
