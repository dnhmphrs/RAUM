use std::error::Error;
use eframe::wgpu;

/// Trait defining a WGPU-based processing pipeline
pub trait Pipeline {
    type Input;
    type Output;
    type Error: Error + 'static;
    
    /// Executes the pipeline on the provided input
    fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Configures the pipeline
    fn configure(&mut self, config: &PipelineConfig) -> Result<(), Self::Error>;
}

/// Configuration for pipelines
pub struct PipelineConfig {
    // Common configuration parameters to be expanded as needed
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
        }
    }
}