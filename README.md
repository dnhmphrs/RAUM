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

# Code Restructuring Plan

## New Directory Structure

```
raum/
├── src/
│   ├── neural/
│   │   ├── mod.rs              # Module exports and shared traits
│   │   ├── hopfield.rs         # Hopfield network implementation (moved from lib.rs)
│   │   └── chip_firing.rs      # Chip Firing Graph implementation (to be added)
│   ├── graphics/
│   │   ├── mod.rs              # Module exports and shared traits
│   │   ├── pipeline.rs         # WGPU pipeline abstractions
│   │   └── renderer.rs         # Rendering utilities
│   ├── ui/
│   │   ├── mod.rs              # Module exports and shared utilities
│   │   ├── app.rs              # Main application wrapper
│   │   ├── windows/
│   │   │   ├── mod.rs          # Window management
│   │   │   ├── hopfield.rs     # Hopfield network window (UI from main.rs)
│   │   │   └── chip_firing.rs  # Chip firing graph window (to be added)
│   │   └── widgets/
│   │       ├── mod.rs          # Common widget exports
│   │       └── grid.rs         # Grid display widget (from existing UI code)
│   ├── lib.rs                  # Library entry point
│   └── main.rs                 # GUI application entry point
```

## Key File Transformations

### 1. Create Module Structure

First, we'll create the necessary directory structure and module files.

### 2. Neural Network Module

#### src/neural/mod.rs
```rust
pub mod hopfield;
// Future: pub mod chip_firing;

/// Common trait for all neural network implementations
pub trait NeuralNetwork {
    type Input;
    type Output;
    type Error;
    
    /// Runs the network forward pass
    fn forward(&self, input: &Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Trains the network on provided data
    fn train(&mut self, data: &[Self::Input]) -> Result<(), Self::Error>;
    
    /// Returns the size or dimension of the network
    fn size(&self) -> usize;
}
```

#### src/neural/hopfield.rs
Move the existing HopfieldNetwork implementation from lib.rs to this file.
- Keep the existing implementation
- Implement the NeuralNetwork trait for HopfieldNetwork

### 3. Graphics Module

#### src/graphics/mod.rs
```rust
pub mod pipeline;
pub mod renderer;

/// Required re-exports
pub use pipeline::Pipeline;
pub use renderer::Renderer;
```

#### src/graphics/pipeline.rs
```rust
/// Trait defining a WGPU-based processing pipeline
pub trait Pipeline {
    type Input;
    type Output;
    type Error;
    
    /// Executes the pipeline on the provided input
    fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Configures the pipeline
    fn configure(&mut self, config: &PipelineConfig) -> Result<(), Self::Error>;
}

/// Configuration for pipelines
pub struct PipelineConfig {
    // Common configuration parameters
    // To be expanded as needed
}
```

#### src/graphics/renderer.rs
```rust
use wgpu;

/// Basic renderer trait
pub trait Renderer {
    /// Initializes the renderer
    fn initialize(&mut self, device: &wgpu::Device) -> Result<(), Box<dyn std::error::Error>>;
    
    /// Renders the current state
    fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView);
}
```

### 4. UI Module

#### src/ui/mod.rs
```rust
pub mod app;
pub mod windows;
pub mod widgets;

// Re-exports
pub use app::RaumApp;
```

#### src/ui/app.rs
```rust
use eframe::egui;

/// Main application wrapper
pub struct RaumApp {
    // State fields
}

impl RaumApp {
    /// Creates a new application instance
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Initialization
        Self {
            // Initialize fields
        }
    }
}

impl eframe::App for RaumApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Main UI code
        // Windows will be managed here
    }
}
```

#### src/ui/windows/mod.rs
```rust
pub mod hopfield;
// Future: pub mod chip_firing;

/// Common trait for application windows
pub trait Window {
    /// Shows the window
    fn show(&mut self, ctx: &eframe::egui::Context, open: &mut bool);
    
    /// Returns the window name
    fn name(&self) -> &str;
}
```

#### src/ui/windows/hopfield.rs
Move the Hopfield UI components from main.rs to this file, restructuring as needed.

#### src/ui/widgets/mod.rs
```rust
pub mod grid;
```

#### src/ui/widgets/grid.rs
Extract the grid drawing functionality from main.rs to create a reusable widget.

### 5. Update Entry Points

#### src/lib.rs
```rust
pub mod neural;
pub mod graphics;
pub mod ui;

// Re-exports
pub use neural::hopfield::HopfieldNetwork;
```

#### src/main.rs
```rust
use eframe;
use raum::ui::RaumApp;

fn main() -> Result<(), eframe::Error> {
    env_logger::init();
    
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([600.0, 400.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "Raum",
        options,
        Box::new(|cc| Box::new(RaumApp::new(cc))),
    )
}
```

## Implementation Steps

1. Create the directory structure
2. Create the module files
3. Move the Hopfield implementation from lib.rs to neural/hopfield.rs
4. Extract UI components into their respective files
5. Update lib.rs and main.rs to use the new structure
6. Ensure the application compiles and runs correctly
7. Prepare for the addition of Chip Firing Graphs