use eframe::egui;
use eframe::egui::TextureHandle;
use std::collections::HashMap;
use std::path::Path;

use crate::ui::windows::{self, Window};

// Helper function to load image for egui
fn load_image_for_ui(path: &Path) -> Result<egui::ColorImage, image::ImageError> {
    let image = image::open(path)?.to_rgba8();
    let size = [image.width() as _, image.height() as _];
    let image_buffer = image.into_raw();
    Ok(egui::ColorImage::from_rgba_unmultiplied(size, &image_buffer))
}

/// Main application structure
pub struct RaumApp {
    /// Collection of windows that can be opened
    windows: HashMap<String, Box<dyn Window>>,
    /// Track which windows are currently open
    window_open_states: HashMap<String, bool>,
    /// Texture handle for the application icon
    icon_texture: Option<TextureHandle>,
}

impl RaumApp {
    /// Creates a new application instance
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Set up custom fonts if needed
        let egui_ctx = &cc.egui_ctx;
        
        // Load the icon texture
        let icon_texture = match load_image_for_ui(Path::new("assets/icon.png")) {
            Ok(image) => Some(egui_ctx.load_texture(
                "app_icon", 
                image, 
                Default::default()
            )),
            Err(e) => {
                eprintln!("Failed to load icon for UI from assets/icon.png: {}", e);
                None
            }
        };

        // Initialize with default windows
        let mut windows: HashMap<String, Box<dyn Window>> = HashMap::new();
        let mut window_open_states: HashMap<String, bool> = HashMap::new();
        
        // Add Hopfield Network window
        let hopfield_window = windows::hopfield::HopfieldWindow::new();
        let window_name_hopfield = hopfield_window.name().to_string();
        windows.insert(window_name_hopfield.clone(), Box::new(hopfield_window));
        window_open_states.insert(window_name_hopfield, false); // Closed by default
        
        // Add Chip Firing Graph window
        let chip_firing_window = windows::chip_firing::ChipFiringWindow::new();
        let window_name_chip = chip_firing_window.name().to_string();
        windows.insert(window_name_chip.clone(), Box::new(chip_firing_window));
        window_open_states.insert(window_name_chip, false); // Closed by default
        
        // Future windows go here
        
        Self {
            windows,
            window_open_states,
            icon_texture,
        }
    }
}

impl eframe::App for RaumApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Top Header Bar ---
        egui::TopBottomPanel::top("main_menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // Display Icon on the left
                if let Some(icon) = &self.icon_texture {
                    ui.add(egui::Image::new(icon).max_height(16.0)); // Adjust size as needed
                }

                ui.menu_button("File", |ui| {
                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    // Add other file options here if needed
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        // TODO: Implement About dialog
                    }
                });
                // Add icon space to the right if desired later
                // ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                //     ui.label("ICON"); // Placeholder
                // });
            });
        });

        // --- Right Sidebar for Controls & Window Toggles ---
        egui::SidePanel::right("config_sidebar")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Configuration");
                ui.separator();

                // Window Toggles
                ui.label("Windows:");
                let mut open_window_names = Vec::new(); // Collect names of windows that should be open
                for (name, is_open) in &mut self.window_open_states {
                    if ui.checkbox(is_open, name).clicked() {
                        // Toggle state handled by checkbox binding
                    }
                    if *is_open {
                        open_window_names.push(name.clone());
                    }
                }
                ui.separator();
                
                // Show config for open windows
                if !open_window_names.is_empty() {
                    ui.label("Settings:");
                    // Use ScrollArea for potentially long configs
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for name in open_window_names {
                            if let Some(window) = self.windows.get_mut(&name) {
                                ui.push_id(name.clone(), |ui| { // Ensure unique IDs for widgets (use cloned name for ID)
                                    let window_name = window.name().to_string(); // Get name before borrow conflict
                                    ui.collapsing(window_name.clone(), |ui| { // Use cloned name for header
                                        window.show_config(ui);    
                                    }).header_response.on_hover_text(format!("Configure {}", window_name)); // Use cloned name for hover
                                });
                                ui.separator(); // Add separator between window configs
                            }
                        }
                    });
                } else {
                    ui.label("Select a window to configure.");
                }
            });

        // --- Individual Windows --- 
        // Iterate through windows and show the content for the open ones in separate egui windows.
        let mut open_window_states = self.window_open_states.clone(); // Clone to avoid borrow issues
        for (name, is_open) in open_window_states.iter_mut() {
            if *is_open {
                if let Some(window) = self.windows.get_mut(name) {
                    // Create an actual egui::Window for the content
                    let window_name = window.name().to_string(); // Get name for title
                    egui::Window::new(&window_name)
                        .open(is_open) // Bind the window's open state to our map
                        .resizable(true)
                        .default_width(400.0) // Give a default size
                        .show(ctx, |ui| {
                            window.show_content(ctx, ui); // Call the content drawing function
                        });
                }
            }
        }
        // Update the original map with potentially changed states (from window closing)
        self.window_open_states = open_window_states;

        // Optional: Add a central panel back if you want something when *no* windows are open
        // egui::CentralPanel::default().show(ctx, |ui| {
        //     ui.vertical_centered(|ui| {
        //         ui.heading("Raum");
        //         ui.label("Open a window from the sidebar.");
        //     });
        // });
    }
}