//! NCP (Neural Circuit Policy) Wiring Example
//!
//! This example demonstrates how to use structured NCP wiring with LTC and CfC networks.
//! NCP creates sparse, layered architectures inspired by biological neural circuits.

use burn::backend::NdArray;
use burn::tensor::Tensor;
use ncps::rnn::{CfC, LTC};
use ncps::wirings::{AutoNCP, NCP, Wiring};

fn main() {
    println!("=== NCP Wiring Example ===\n");

    type Backend = NdArray<f32>;
    let device = Default::default();

    // Example 1: Using AutoNCP (convenience constructor)
    println!("Example 1: AutoNCP Wiring with LTC");
    println!("Configuring NCP with 64 total neurons, 8 motor outputs");

    // AutoNCP automatically configures the architecture
    // Units: 64, Output size: 8, Sparsity: 0.65
    let wiring = AutoNCP::new(64, 8, 0.65, 22222);

    let ltc = LTC::<Backend>::new(16, wiring, &device);

    let input = Tensor::<Backend, 3>::random(
        [2, 20, 16],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let (output, _state) = ltc.forward(input, None, None);

    println!("  Architecture: 64 neurons, 8 motor outputs");
    println!("  Input shape:  [2, 20, 16]");
    println!("  Output shape: {:?}", output.dims());
    println!("  (Output is 8 motor neurons only)");
    println!();

    // Example 2: Manual NCP configuration
    println!("Example 2: Custom NCP Configuration");

    // Manually specify each layer
    let wiring_manual = NCP::new(
        12,    // inter_neurons (layer 2)
        8,     // command_neurons (layer 3)
        4,     // motor_neurons (layer 4 = outputs)
        5,     // sensory_fanout (connections from input to inter)
        4,     // inter_fanout (connections from inter to command)
        3,     // recurrent_command_synapses (self-connections in command)
        4,     // motor_fanin (connections from command to motor)
        12345, // random seed
    );

    let cfc = CfC::<Backend>::new(20, wiring_manual.units(), &device);

    let input2 = Tensor::<Backend, 3>::random(
        [4, 15, 20],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let (output2, _state2) = cfc.forward(input2, None, None);

    println!("  Architecture:");
    println!("    - 20 sensory inputs");
    println!("    - 12 inter neurons");
    println!("    - 8 command neurons");
    println!("    - 4 motor outputs");
    println!("  Total neurons: {}", 12 + 8 + 4);
    println!("  Input shape:  [4, 15, 20]");
    println!("  Output shape: {:?}", output2.dims());
    println!();

    // Example 3: Sparse vs dense connectivity
    println!("Example 3: Connectivity Comparison");

    // AutoNCP with different motor sizes
    // Note: wiring must be built before synapse_count() returns meaningful values
    for motor_size in [2, 4, 8, 16] {
        let mut wiring = AutoNCP::new(64, motor_size, 0.5, 42);
        wiring.build(16); // Build with 16 input features to populate adjacency matrices
        let synapse_count = wiring.synapse_count();

        println!(
            "  NCP(64 total, {} motor): {} internal synapses",
            motor_size, synapse_count
        );
    }
    println!();

    // Example 4: Network with mixed memory
    println!("Example 4: LTC with Mixed Memory (LSTM)");

    let wiring_mm = AutoNCP::new(32, 8, 0.5, 12345);
    let ltc_mm = LTC::<Backend>::new(12, wiring_mm, &device).with_mixed_memory(true, &device);

    let input_mm = Tensor::<Backend, 3>::random(
        [2, 25, 12],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    // For mixed memory, use forward_mixed which returns (output, (h, c))
    let (output_mm, (h_n, c_n)) = ltc_mm.forward_mixed(input_mm, None, None);

    println!("  Architecture: NCP(32 neurons, 8 motor) + LSTM");
    println!("  Input shape:  [2, 25, 12]");
    println!("  Output shape: {:?}", output_mm.dims());
    println!("  Final hidden shape: {:?}", h_n.dims());
    println!("  Final cell shape:   {:?}", c_n.dims());
    println!();

    println!("=== NCP Examples completed! ===");
}
