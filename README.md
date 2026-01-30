# NCPS - Neural Circuit Policies (Rust)

A Rust port of the [NCPS library](https://github.com/mlech26l/ncps) using the [Burn](https://burn.dev) deep learning framework.

Neural Circuit Policies (NCPs) are sparse recurrent neural networks inspired by the nervous system of *C. elegans*, designed for interpretable and efficient sequence modeling.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ncps = { version = "0.1", features = ["ndarray"] }
burn = { version = "0.16", features = ["ndarray"] }
```

### Backend Features

- `ndarray` (default) - CPU backend
- `wgpu` - GPU backend via WebGPU
- `candle` - Alternative GPU backend

## Quick Start

```rust
use burn::backend::NdArray;
use burn::tensor::Tensor;
use ncps::prelude::*;

type Backend = NdArray<f32>;

fn main() {
    let device = Default::default();

    // Create NCP wiring: 64 neurons, 8 outputs, 50% sparsity
    let wiring = AutoNCP::new(64, 8, 0.5, 42);

    // Create LTC RNN layer
    let ltc = LTC::<Backend>::new(12, wiring, &device);

    // Process sequence [batch=2, seq_len=10, features=12]
    let input = Tensor::<Backend, 3>::zeros([2, 10, 12], &device);
    let (output, final_state) = ltc.forward(input, None, None);

    println!("Output: {:?}", output.dims());  // [2, 10, 8]
}
```

## Core Components

### Wiring Configurations

| Type | Description |
|------|-------------|
| `FullyConnected` | Dense connectivity (all-to-all) |
| `NCP` | Neural Circuit Policy - 4-layer biologically-inspired architecture |
| `AutoNCP` | Convenience wrapper for NCP with automatic sizing |
| `Random` | Sparse random connectivity |

```rust
// NCP: inter_neurons, command_neurons, motor_neurons, fanout params...
let ncp = NCP::new(20, 10, 8, 4, 4, 3, 4, 42);

// AutoNCP: total_units, output_size, sparsity_level, seed
let auto_ncp = AutoNCP::new(64, 8, 0.5, 42);

// FullyConnected: units, output_dim, seed, self_connections
let fc = FullyConnected::new(32, Some(8), 1234, true);

// Random: units, output_dim, sparsity_level, seed
let random = Random::new(32, Some(8), 0.5, 1234);
```

### Cell Types

| Cell | Description |
|------|-------------|
| `LTCCell` | Liquid Time-Constant - ODE-based continuous-time dynamics |
| `CfCCell` | Closed-form Continuous-time - fast approximation of LTC |
| `WiredCfCCell` | Multi-layer CfC respecting NCP wiring structure |
| `LSTMCell` | Standard LSTM for mixed memory mode |

### RNN Layers

| Layer | Description |
|-------|-------------|
| `LTC` | Full sequence processing with LTC cells |
| `CfC` | Full sequence processing with CfC cells |

## Features

### Input/Output Mapping (LTC)

```rust
use ncps::cells::{LTCCell, MappingMode};

let cell = LTCCell::<Backend>::new(&wiring, Some(16), &device)
    .with_input_mapping(MappingMode::Affine, &device)   // y = w*x + b
    .with_output_mapping(MappingMode::Linear, &device); // y = w*x
```

### CfC Modes

```rust
use ncps::cells::{CfCCell, CfcMode};

let cell = CfCCell::<Backend>::new(20, 50, &device)
    .with_mode(CfcMode::Default)  // Gated interpolation (default)
    .with_mode(CfcMode::Pure)     // Direct ODE solution
    .with_mode(CfcMode::NoGate);  // Simplified gating
```

### Mixed Memory (LSTM Augmentation)

```rust
let ltc = LTC::<Backend>::new(12, wiring, &device)
    .with_mixed_memory(true, &device);

// Forward returns (output, (h_state, c_state))
let (output, (h, c)) = ltc.forward_mixed(input, None, None);
```

### Variable Time Steps

```rust
// Custom time intervals per timestep
let timespans = Tensor::<Backend, 2>::full([batch, seq_len], 0.1, &device);
let (output, state) = ltc.forward(input, None, Some(timespans));
```

### Batch & Sequence Options

```rust
let ltc = LTC::<Backend>::new(12, wiring, &device)
    .with_batch_first(true)       // Input: [batch, seq, features] (default)
    .with_return_sequences(true); // Return all timesteps (default)

// Or return only last timestep
let ltc = ltc.with_return_sequences(false);
```

## Examples

Run the included examples:

```bash
cargo run --example basic       # Basic CfC usage
cargo run --example ncp_wiring  # NCP wiring configurations
cargo run --example save_load   # Model serialization
```

## Architecture

```
ncps/
├── cells/
│   ├── ltc_cell.rs      # LTC with ODE solver, sparsity masks
│   ├── cfc_cell.rs      # CfC with 3 modes, sparsity support
│   ├── wired_cfc_cell.rs # Multi-layer CfC for NCP
│   └── lstm_cell.rs     # LSTM for mixed memory
├── rnn/
│   ├── ltc.rs           # LTC sequence processing
│   └── cfc.rs           # CfC sequence processing
├── wirings/
│   ├── base.rs          # Wiring trait, FullyConnected
│   ├── ncp.rs           # NCP, AutoNCP
│   └── random.rs        # Random sparse wiring
└── activation.rs        # LeCun tanh activation
```

## Key Differences from Python

| Feature | Python | Rust |
|---------|--------|------|
| Framework | PyTorch/TensorFlow | Burn |
| Wiring param | `units` (int or Wiring) | Always `impl Wiring` |
| Builder pattern | kwargs | `.with_*()` methods |
| Backend | Runtime selection | Compile-time features |

## References

- [Liquid Time-constant Networks](https://ojs.aaai.org/index.php/AAAI/article/view/16936) (AAAI 2021)
- [Closed-form Continuous-time Neural Networks](https://www.nature.com/articles/s42256-022-00556-7) (Nature Machine Intelligence 2022)
- [Neural Circuit Policies](https://publik.tuwien.ac.at/files/publik_292280.pdf) (CoRL 2020)

## License

Apache-2.0
# ncps-rust
