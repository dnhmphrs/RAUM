# Raum

Raum is a spatial processing system built in Rust, designed to provide powerful abstractions for neural networks and graphics pipelines.

## Vision

Raum aims to be a flexible, modular library that enables:

1. Neural network implementations with visualization capabilities
2. WGPU-powered graphics pipelines for mathematical visualization
3. Cross-language usage through FFI bindings for C++ and Python
4. Interactive exploration of geometric and analytical mathematics

## Architecture

The project is structured into several key components:

### Core Components

- **Neural Networks**: Implementations of various neural network types
  - Hopfield Networks (associative memory)
  - Chip Firing Graphs (discrete dynamical systems)
  - More network types planned for future expansion

- **Graphics**: WGPU-based rendering pipelines
  - Configurable shaders for mathematical visualization
  - Efficient GPU-accelerated computation

- **UI**: Modular egui-based interface
  - Window-based application structure
  - Interactive visualization controls

### Interface Options

- **GUI Application**: Interactive exploration and visualization
- **Library**: Use within other Rust applications
- **FFI**: C++ and Python bindings for cross-language usage

## Current Status

The project currently has:
- An implementation of Hopfield networks
- A basic GUI for Hopfield network visualization and interaction
- Foundation for WGPU rendering

## Near-Term Roadmap

1. **Code Restructuring**:
   - Separate neural network implementations
   - Modularize UI components
   - Create abstractions for rendering pipelines

2. **Feature Expansion**:
   - Add Chip Firing Graph implementation
   - Create window-based UI infrastructure
   - Implement basic WGPU rendering pipelines

3. **Documentation**:
   - API documentation
   - Usage examples
   - Mathematical foundations

## Long-Term Goals

- Browser-based WGPU rendering
- GPU-accelerated projections for higher-dimensional mathematics
- Comprehensive neural network library with visualization
- Document generation for mathematical exploration

## Project Structure

```
raum/
├── src/
│   ├── neural/                 # Neural network implementations
│   │   ├── hopfield.rs         # Hopfield network implementation
│   │   └── chip_firing.rs      # Chip firing graph implementation
│   ├── graphics/               # Graphics and rendering
│   │   ├── pipeline.rs         # WGPU pipeline abstractions
│   │   └── renderer.rs         # Rendering utilities
│   ├── ui/                     # UI components
│   │   ├── app.rs              # Main application
│   │   ├── windows/            # Application windows
│   │   │   ├── hopfield.rs     # Hopfield network window
│   │   │   └── chip_firing.rs  # Chip firing graph window
│   │   └── widgets/            # Reusable UI widgets
│   ├── ffi/                    # Foreign Function Interface
│   │   ├── c.rs                # C bindings
│   │   └── python.rs           # Python bindings
│   ├── lib.rs                  # Library entry point
│   └── main.rs                 # GUI application entry point
└── examples/                   # Example applications
```

## Usage

*Documentation coming soon*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

*License information to be added*