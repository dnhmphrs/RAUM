use std::error::Error;
use eframe::wgpu;

/// Basic renderer trait
pub trait Renderer {
    /// Initializes the renderer with the provided device
    fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), Box<dyn Error>>;
    
    /// Renders the current state to the provided view
    fn render(
        &self, 
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder, 
        view: &wgpu::TextureView
    );
    
    /// Updates renderer state (e.g., with new data)
    fn update(&mut self, queue: &wgpu::Queue, data: &[u8]) -> Result<(), Box<dyn Error>>;
    
    /// Resizes the renderer's resources
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32);
}