pub mod hopfield;
// Future: pub mod chip_firing;

use eframe::egui;

/// Common trait for application windows
pub trait Window {
    /// Draws the main content of the window
    fn show_content(&mut self, ctx: &egui::Context, ui: &mut egui::Ui);
    
    /// Draws the configuration controls for the window
    fn show_config(&mut self, ui: &mut egui::Ui);
    
    /// Returns the window name
    fn name(&self) -> &str;
}