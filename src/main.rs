#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use hopfield::{HopfieldNetwork}; // Import from your library
use rand::Rng; // Import rand for noise generation
use rand::rngs::ThreadRng; // For storing rng state
use egui_plot::{Line, Plot, PlotPoints}; // Import egui_plot items
use rusttype::{point, Font, Scale};

// Removed unused constants
// const WIDTH: usize = 48;
// const HEIGHT: usize = 48;
// const NUM_NEURONS: usize = WIDTH * HEIGHT; // 2304

// Generate patterns of specified size for given characters
fn get_patterns(grid_size: usize, characters: &[char]) -> Vec<(char, Vec<f64>)> {
    // --- Configuration ---
    let font_path = "font.otf"; // <---- CHANGE THIS if your font file name is different
    // let characters = "ABC"; // Now passed as parameter
    let reference_pixel_height = 100.0; // Render large initially to find bounds accurately
    let threshold = 0.5; // Coverage threshold to make a pixel 'on' (1.0)
    let target_width = grid_size as f32;
    let target_height = grid_size as f32;
    let num_neurons_dynamic = grid_size * grid_size; // Use dynamic size
    // ---------------------

    println!(
        "Attempting to load font: {}, Grid size: {}, Chars: {:?}",
        font_path, grid_size, characters
    );
    let font_data = match std::fs::read(font_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!(
                "Error loading font file '{}': {}. Using fallback patterns.",
                font_path, e
            );
            // Fallback to basic patterns if font fails
            let pattern_a = vec![-1.0; num_neurons_dynamic]; // Use dynamic size
            let pattern_b = vec![1.0; num_neurons_dynamic]; // Use dynamic size
            if characters.is_empty() {
                return vec![];
            } else if characters.len() == 1 {
                 return vec![(characters[0], pattern_a)];
            } else {
                 return vec![(characters[0], pattern_a), (characters[1], pattern_b)];
            }
           
        }
    };

    let font = match Font::try_from_vec(font_data) {
        Some(f) => f,
        None => {
            eprintln!(
                "Error parsing font file '{}'. Using fallback patterns.",
                font_path
            );
             let pattern_a = vec![-1.0; num_neurons_dynamic]; // Use dynamic size
            let pattern_b = vec![1.0; num_neurons_dynamic]; // Use dynamic size
             if characters.is_empty() {
                return vec![];
            } else if characters.len() == 1 {
                 return vec![(characters[0], pattern_a)];
            } else {
                 return vec![(characters[0], pattern_a), (characters[1], pattern_b)];
            }
        }
    };

    let mut patterns = Vec::with_capacity(characters.len());

    for char_code in characters.iter() {
        let mut pattern = vec![-1.0; num_neurons_dynamic]; // Use dynamic size

        // --- Step 1: Render large at reference size to get bounds ---
        let large_scale = Scale::uniform(reference_pixel_height);
        let large_glyph = font.glyph(*char_code).scaled(large_scale);
        let positioned_large_glyph = large_glyph.positioned(point(0.0, 0.0));

        if let Some(large_bb) = positioned_large_glyph.pixel_bounding_box() {
            let bb_width = large_bb.width() as f32;
            let bb_height = large_bb.height() as f32;

            if bb_width > 0.0 && bb_height > 0.0 { // Avoid division by zero for empty glyphs
                 // --- Step 2: Calculate final scale to fit 48x48 ---
                let scale_x = target_width / bb_width;
                let scale_y = target_height / bb_height;
                let final_scale_factor = scale_x.min(scale_y); // Preserve aspect ratio
                let final_pixel_height = reference_pixel_height * final_scale_factor;
                let final_scale = Scale::uniform(final_pixel_height);

                // --- Step 3: Re-render glyph at final scale ---
                let final_glyph = font.glyph(*char_code).scaled(final_scale);
                 // Position at origin to measure bounds for centering
                let positioned_final_glyph = final_glyph.positioned(point(0.0, 0.0));

                if let Some(final_bb) = positioned_final_glyph.pixel_bounding_box() {
                    // --- Step 4: Calculate centering offset for the final rendering ---
                    let final_bb_width = final_bb.width() as f32;
                    let final_bb_height = final_bb.height() as f32;
                    let target_x = ((target_width - final_bb_width) / 2.0).round();
                    let target_y = ((target_height - final_bb_height) / 2.0).round();

                    // --- Step 5: Draw final glyph at origin & copy pixels to centered position ---
                    positioned_final_glyph.draw(|px, py, v| {
                        let final_pattern_x = px as i32 + target_x as i32;
                        let final_pattern_y = py as i32 + target_y as i32;

                        if final_pattern_x >= 0
                            && final_pattern_x < grid_size as i32
                            && final_pattern_y >= 0
                            && final_pattern_y < grid_size as i32
                        {
                            if v > threshold {
                                let index = final_pattern_y as usize * grid_size
                                    + final_pattern_x as usize;
                                pattern[index] = 1.0;
                            }
                        }
                    });
                } // End if final_bb
            } // End if bb_width/height > 0
        } // End if large_bb

        patterns.push((*char_code, pattern));
    }
    println!("Successfully generated {} patterns from font.", patterns.len());
    patterns
}

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([600.0, 400.0]), // Increased size
        ..Default::default()
    };
    eframe::run_native(
        "Hopfield Network GUI", // Changed title
        options,
        Box::new(|_cc| Ok(Box::<MyApp>::default())),
    )
}

struct MyApp {
    network: Option<HopfieldNetwork>,
    // Store all patterns generated for the current grid size (A-Z usually)
    all_generated_patterns: Vec<(char, Vec<f64>)>, 
    // Store the subset of patterns actually used for training
    patterns: Vec<Vec<f64>>,
    // Store the characters corresponding to the `patterns` used for training
    trained_chars: Vec<char>,
    // UI State: Grid Size
    current_grid_size: usize,
    // UI State: Characters available for selection (fixed A-Z for now)
    available_chars: Vec<char>,
    // UI State: Which characters are selected for training (Indices into available_chars)
    selected_indices_for_training: std::collections::HashSet<usize>,
    // UI State: Index of the pattern selected as the base for the input_state
    selected_pattern_index_for_input: Option<usize>,
    noise_level: f32, // 0.0 to 1.0
    input_state: Vec<f64>,
    // Store the sequence of states during convergence
    output_states: Option<Vec<Vec<f64>>>, 
    // Store the corresponding energy for each state in output_states
    energy_history: Option<Vec<f64>>, 
    // Which iteration to display from output_states
    display_iteration: Option<usize>,
    iterations: Option<usize>,
    error_message: Option<String>,
    max_iterations: usize,
    // Store RNG for async updates
    rng: ThreadRng,
     // UI State: Update mode
    update_mode: UpdateMode,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum UpdateMode {
    Synchronous,
    Asynchronous,
}

impl Default for MyApp {
    fn default() -> Self {
        let initial_grid_size = 48;
        let available_chars: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect();
        // Default to training on 'A', 'B', 'C', 'D', 'E'
        let initial_selected_indices: std::collections::HashSet<usize> = (0..5).collect();
        
        // Generate initial set of all patterns (A-Z) at default size
        let all_generated_patterns = get_patterns(initial_grid_size, &available_chars);

        // Create the initial training subset based on default selection
        let (patterns, trained_chars) = filter_patterns(&all_generated_patterns, &initial_selected_indices);

        // Select the first pattern of the *training subset* as the initial input
        let initial_input = patterns.get(0).cloned().unwrap_or_else(|| vec![0.0; initial_grid_size * initial_grid_size]);
        let initial_selected_pattern_index = if patterns.is_empty() { None } else { Some(0) };

        Self {
            network: None,
            all_generated_patterns,
            patterns, 
            trained_chars,
            current_grid_size: initial_grid_size,
            available_chars,
            selected_indices_for_training: initial_selected_indices,
            selected_pattern_index_for_input: initial_selected_pattern_index,
            noise_level: 0.0,
            input_state: initial_input,
            output_states: None,
            energy_history: None,
            display_iteration: None,
            iterations: None,
            error_message: None,
            max_iterations: 100,
            rng: rand::thread_rng(), // Initialize RNG
            update_mode: UpdateMode::Synchronous, // Default to Synchronous
        }
    }
}

// Helper function to filter the full pattern set based on selected indices
fn filter_patterns(all_patterns: &[(char, Vec<f64>)], selected_indices: &std::collections::HashSet<usize>) -> (Vec<Vec<f64>>, Vec<char>) {
    let mut patterns_subset = Vec::new();
    let mut chars_subset = Vec::new();
    for (idx, (char, pattern)) in all_patterns.iter().enumerate() {
        if selected_indices.contains(&idx) {
            patterns_subset.push(pattern.clone());
            chars_subset.push(*char);
        }
    }
    (patterns_subset, chars_subset)
}

/// Applies noise to a state vector by flipping bits.
/// `noise_level` is the probability (0.0 to 1.0) that any given bit is flipped.
fn apply_noise(state: &[f64], noise_level: f32) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    state
        .iter()
        .map(|&val| {
            if rng.gen::<f32>() < noise_level {
                -val // Flip the bit
            } else {
                val // Keep the bit
            }
        })
        .collect()
}

/// Draws a Hopfield network state vector as a grid.
fn draw_grid(ui: &mut egui::Ui, state: &[f64], width: usize, height: usize, cell_size: f32) {
    // Prevent drawing if state is empty or incorrect size (might happen during initialization)
    if state.len() != width * height {
        ui.label("Invalid state for grid display");
        return;
    }
    // Calculate the total size needed for the grid
    let grid_size = egui::vec2(width as f32 * cell_size, height as f32 * cell_size);
    // Allocate space for the grid
    let (response, painter) = ui.allocate_painter(grid_size, egui::Sense::hover());
    let rect = response.rect;

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let cell_state = state.get(index).copied().unwrap_or(0.0); // Default to 0.0 if out of bounds (shouldn't happen)

            let cell_color = if cell_state == 1.0 {
                egui::Color32::BLACK
            } else if cell_state == -1.0 {
                egui::Color32::WHITE
            } else {
                egui::Color32::GRAY // Should not happen with valid states
            };

            let cell_top_left = rect.min + egui::vec2(x as f32 * cell_size, y as f32 * cell_size);
            let cell_rect = egui::Rect::from_min_size(cell_top_left, egui::vec2(cell_size, cell_size));

            painter.rect_filled(cell_rect, 0.0, cell_color);
            // Add a border for clarity
            painter.rect_stroke(cell_rect, 0.0, egui::Stroke::new(1.0, egui::Color32::DARK_GRAY));
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Side Panel for Patterns and Controls ---
        egui::SidePanel::left("control_panel").show(ctx, |ui| {
            // --- Top Section Controls ---
            ui.heading("Controls");
            ui.separator();
            // Grid Size
            ui.label("Grid Size (N x N):");
            let mut grid_size_mut = self.current_grid_size;
            let grid_slider = ui.add(egui::DragValue::new(&mut grid_size_mut)
                .speed(1.0)
                .range(8..=64) // Set a reasonable range
                .clamp_to_range(true));

            if grid_slider.changed() {
                if grid_size_mut != self.current_grid_size {
                    println!("Grid size changed to: {}", grid_size_mut);
                    self.current_grid_size = grid_size_mut;
                    // Regenerate all patterns at the new size
                    self.all_generated_patterns = get_patterns(self.current_grid_size, &self.available_chars);
                    // Update the training subset
                    let (new_patterns, new_trained_chars) = filter_patterns(&self.all_generated_patterns, &self.selected_indices_for_training);
                    self.patterns = new_patterns;
                    self.trained_chars = new_trained_chars;
                    // Reset network and output
                    self.network = None;
                    self.output_states = None;
                    self.energy_history = None;
                    self.iterations = None;
                    self.display_iteration = None;
                    self.selected_pattern_index_for_input = if self.patterns.is_empty() { None } else { Some(0) };
                    // Update input state based on new size and first selected pattern
                    self.input_state = self.patterns.get(0).cloned().unwrap_or_else(|| vec![0.0; self.current_grid_size * self.current_grid_size]);
                    if let Some(p) = self.patterns.get(0) { // Re-apply noise if a pattern exists
                        self.input_state = apply_noise(p, self.noise_level);
                    }
                }
            }
            ui.separator();

            // Character Selection
            ui.label("Select Characters for Training:");
            let mut selection_changed = false;
            egui::ScrollArea::vertical()
                .id_source("training_char_scroll")
                .max_height(150.0)
                .show(ui, |ui| {
                    egui::Grid::new("training_char_grid")
                        .num_columns(5) // Adjust number of columns as desired
                        .spacing([5.0, 5.0])
                        .show(ui, |ui| {
                             let mut col_count = 0;
                             for (idx, &char_code) in self.available_chars.iter().enumerate() {
                                let mut is_selected = self.selected_indices_for_training.contains(&idx);
                                if ui.checkbox(&mut is_selected, char_code.to_string()).changed() {
                                    if is_selected {
                                        self.selected_indices_for_training.insert(idx);
                                    } else {
                                        self.selected_indices_for_training.remove(&idx);
                                    }
                                    selection_changed = true;
                                }
                                col_count += 1;
                                if col_count % 5 == 0 { // End row after N columns
                                    ui.end_row();
                                }
                            }
                        });
                 });

            if selection_changed {
                println!("Training selection changed: {:?}", self.selected_indices_for_training);
                // Update the training subset
                 let (new_patterns, new_trained_chars) = filter_patterns(&self.all_generated_patterns, &self.selected_indices_for_training);
                self.patterns = new_patterns;
                self.trained_chars = new_trained_chars;
                 // Reset network and output
                self.network = None;
                self.output_states = None;
                self.energy_history = None;
                self.iterations = None;
                self.display_iteration = None;
                // Reset selected input pattern index if it's no longer valid
                let current_input_char = self.selected_pattern_index_for_input.and_then(|i| self.trained_chars.get(i));
                self.selected_pattern_index_for_input = if self.patterns.is_empty() { 
                    None 
                } else if current_input_char.is_none() || !self.trained_chars.contains(current_input_char.unwrap()) {
                    Some(0) // Default to first if previous invalid or none exists
                } else {
                     self.selected_pattern_index_for_input // Keep if still valid
                };
                 // Update input state based on (potentially new) first selected pattern
                self.input_state = self.patterns.get(self.selected_pattern_index_for_input.unwrap_or(0))
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; self.current_grid_size * self.current_grid_size]);
                 if let Some(idx) = self.selected_pattern_index_for_input {
                    if let Some(p) = self.patterns.get(idx) { // Re-apply noise if a pattern exists
                        self.input_state = apply_noise(p, self.noise_level);
                    }
                 }
            }

            ui.separator();

            // Train Button
            if ui.button("Train Network").clicked() {
                if self.patterns.is_empty() {
                    self.error_message = Some("Cannot train: No patterns selected.".to_string());
                } else {
                    self.error_message = None;
                    self.output_states = None;
                    self.energy_history = None;
                    self.iterations = None;
                    self.display_iteration = None;
                    // Create network with current dynamic size
                    let mut net = HopfieldNetwork::new(self.current_grid_size * self.current_grid_size);
                    match net.train(&self.patterns) {
                        Ok(_) => {
                            self.network = Some(net);
                            println!("Network trained successfully on {} patterns.", self.patterns.len());
                        }
                        Err(e) => {
                            self.network = None;
                            self.error_message = Some(format!("Training Error: {}", e));
                        }
                    }
                }
            }

            ui.separator();

            // --- Middle Section (Select Initial Pattern) ---
            ui.vertical_centered(|ui| {
                ui.label("Select Initial Pattern (from trained set):");
            });
            // Calculate available height for the middle section, leaving space for bottom controls
            let bottom_controls_estimated_height = 200.0;
            let middle_scroll_max_height = (ui.available_height() - bottom_controls_estimated_height).max(100.0);

            egui::ScrollArea::vertical()
                .id_source("input_pattern_scroll")
                .max_height(middle_scroll_max_height) // Explicitly set max height
                .show(ui, |ui| {
                    // Use a grid layout for better spacing
                    egui::Grid::new("input_pattern_grid")
                        .num_columns(2)
                        .spacing([10.0, 5.0])
                        .show(ui, |ui| {
                             for (subset_idx, &char_code) in self.trained_chars.iter().enumerate() {
                                let label_text = char_code.to_string();
                                let is_selected = self.selected_pattern_index_for_input == Some(subset_idx);
                                // Column 1: Selectable Label
                                if ui.selectable_label(is_selected, &label_text).clicked() {
                                    // Update input selection immediately
                                    if self.selected_pattern_index_for_input != Some(subset_idx) {
                                        self.selected_pattern_index_for_input = Some(subset_idx);
                                        // Update input state based on new selection
                                        if let Some(selected_pattern) = self.patterns.get(subset_idx) {
                                            if selected_pattern.len() == self.current_grid_size * self.current_grid_size {
                                                self.input_state = apply_noise(selected_pattern, self.noise_level);
                                                self.output_states = None; 
                                                self.energy_history = None;
                                                self.display_iteration = None;
                                                self.iterations = None;
                                            } else {
                                                self.error_message = Some("Input pattern update error: Size mismatch.".to_string());
                                            }
                                        }
                                    }
                                }
                                // Column 2: Preview Grid
                                if let Some(pattern) = self.patterns.get(subset_idx) {
                                    if pattern.len() == self.current_grid_size * self.current_grid_size {
                                       // Increase cell size for preview visibility
                                       draw_grid(ui, pattern, self.current_grid_size, self.current_grid_size, 2.0); // Use 2.0 cell size
                                    } else {
                                       ui.label("(Invalid preview size)");
                                    }
                                } else {
                                     ui.label("(No preview)");
                                }
                                ui.end_row();
                            }
                        });
                 });

            // --- Bottom Section Controls (Draw AFTER the ScrollArea) ---
            ui.separator();
            // Noise Control
            ui.label("Noise Level:");
            let noise_slider = ui.add(egui::Slider::new(&mut self.noise_level, 0.0..=1.0).text("Noise"));
            if noise_slider.changed() {
                 if let Some(idx) = self.selected_pattern_index_for_input {
                     if let Some(selected_pattern) = self.patterns.get(idx) {
                          if selected_pattern.len() == self.current_grid_size * self.current_grid_size {
                             self.input_state = apply_noise(selected_pattern, self.noise_level);
                             self.output_states = None; // Clear output sequence
                             self.energy_history = None; // Clear energy history
                             self.display_iteration = None;
                             self.iterations = None;
                          } else {
                             self.error_message = Some("Noise application error: Size mismatch.".to_string());
                         }
                     }
                 }
            }
            ui.separator();
            // Update Mode Selection
            ui.label("Update Mode:");
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.update_mode, UpdateMode::Asynchronous, "Asynchronous"); // Draw right first
                ui.radio_value(&mut self.update_mode, UpdateMode::Synchronous, "Synchronous");
            });
            ui.separator();
            // Run Controls
            ui.label("Max Iterations:");
             ui.add(egui::DragValue::new(&mut self.max_iterations).speed(1.0).range(1..=1000));
            ui.separator();
            if ui.add_enabled(self.network.is_some(), egui::Button::new("Run Network")).clicked() {
                 if self.input_state.len() != self.current_grid_size * self.current_grid_size {
                     self.error_message = Some("Cannot run: Input state size mismatch.".to_string());
                 } else {
                     self.error_message = None;
                     if let Some(net) = &self.network {
                        // Check network size matches current grid size before running
                        if net.size() != self.current_grid_size * self.current_grid_size {
                            self.error_message = Some("Cannot run: Network size does not match current grid size. Retrain network.".to_string());
                        } else {
                             // Call appropriate run method based on mode
                             let run_result = match self.update_mode {
                                UpdateMode::Synchronous => {
                                    net.run(&self.input_state, self.max_iterations)
                                }
                                UpdateMode::Asynchronous => {
                                    // Pass the mutable rng
                                    net.run_async(&self.input_state, self.max_iterations, &mut self.rng)
                                }
                             };

                             match run_result {
                                Ok((states_history, iters)) => {
                                    println!(
                                        "Network run completed. Iterations: {}. States: {}",
                                        iters,
                                        states_history.len()
                                    );
                                    // Calculate energy for each state
                                    let energies: Result<Vec<f64>, _> = states_history
                                        .iter()
                                        .map(|state| net.energy(state))
                                        .collect();

                                    match energies {
                                        Ok(energy_values) => {
                                             // Assign the whole history
                                            self.output_states = Some(states_history.clone()); 
                                            self.energy_history = Some(energy_values);
                                            self.iterations = Some(iters);
                                            // Default view to the last iteration
                                            self.display_iteration = Some(states_history.len().saturating_sub(1)); 
                                        }
                                        Err(e) => {
                                            // Handle energy calculation error
                                            self.output_states = None;
                                            self.energy_history = None;
                                            self.iterations = None;
                                            self.error_message = Some(format!("Energy Calc Error: {}", e));
                                        }
                                    }
                                }
                                Err(e) => {
                                    self.output_states = None;
                                    self.energy_history = None; // Clear energy history on run error
                                    self.iterations = None;
                                    self.error_message = Some(format!("Runtime Error: {}", e));
                                }
                             } // End match run_result
                        } // End else network size check
                    } // End if let Some(net)
                 } // End else input state check
            } // End if button clicked
            ui.separator();
            // Formulae and Variables
            egui::CollapsingHeader::new("Info & Formulae")
                .id_source("info_collapse")
                .show(ui, |ui| {
                ui.label(format!("Grid Size: {}x{}", self.current_grid_size, self.current_grid_size));
                ui.label(format!("Neurons: {}", self.current_grid_size * self.current_grid_size));
                ui.label(format!("Stored Patterns: {}", self.patterns.len()));
                ui.label(format!("Max Iterations: {}", self.max_iterations));
                ui.separator();
                ui.label("Update Rule (simplified): Sᵢ(t+1) = sgn( Σⱼ Wᵢⱼ Sⱼ(t) ) ");
                ui.label("Learning Rule (Hebbian): Wᵢⱼ = Σₚ ξᵢᵖ ξⱼᵖ  (for i ≠ j, Wᵢᵢ = 0)");
                ui.label("Energy Function: E = -½ Σᵢ Σⱼ Wᵢⱼ Sᵢ Sⱼ (for i ≠ j)");
            });
            // Display Error Messages
            if let Some(err) = &self.error_message {
                ui.separator();
                ui.colored_label(egui::Color32::RED, err);
            }
        });

        // --- Central Panel for Input/Output Display & Energy Plot ---
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Network State & Energy");
            ui.separator();

            // Top part: Target | Input | Output Grids
            ui.columns(3, |columns| {
                // Column 1: Target Pattern
                columns[0].vertical_centered(|ui| {
                    ui.label("Target Pattern");
                    ui.separator();
                    if let Some(idx) = self.selected_pattern_index_for_input {
                        if let Some(pattern) = self.patterns.get(idx) {
                            // Use current grid size for drawing
                            if pattern.len() == self.current_grid_size * self.current_grid_size {
                                draw_grid(ui, pattern, self.current_grid_size, self.current_grid_size, 4.0); 
                                // Show character label from the trained set
                                let label_text = self.trained_chars.get(idx).map(|&c| c.to_string()).unwrap_or_else(|| "?".to_string());
                                ui.label(label_text);
                            } else {
                                ui.label("(Invalid pattern size)");
                            }
                        } else {
                            ui.label("(Pattern not selected)");
                        }
                    } else {
                         ui.label("(Pattern not selected)");
                    }
                });

                // Column 2: Input State (with noise)
                columns[1].vertical_centered(|ui| {
                    ui.label("Input State");
                    ui.separator();
                    // Use current grid size for drawing
                    if self.input_state.len() == self.current_grid_size * self.current_grid_size {
                         draw_grid(ui, &self.input_state, self.current_grid_size, self.current_grid_size, 4.0); 
                    } else {
                         ui.label("(Invalid input state size)");
                    }
                    ui.label(format!("Noise: {:.2}", self.noise_level));
                    if let Some(idx) = self.selected_pattern_index_for_input {
                        // Show character label from the trained set
                         let label_text = self.trained_chars.get(idx).map(|&c| c.to_string()).unwrap_or_else(|| "?".to_string());
                        ui.label(format!("Based on {}", label_text));
                    }
                 });

                 // Column 3: Output State (Iteration Viewer)
                 columns[2].vertical_centered(|ui|{
                    ui.label("Output State (Iteration Viewer)");
                    ui.separator();
                    if let Some(states) = &self.output_states {
                        // Get the state to display based on the slider
                        let iteration_to_display = self.display_iteration.unwrap_or(0).min(states.len().saturating_sub(1));
                        if let Some(output) = states.get(iteration_to_display) {
                             // Use current grid size for drawing
                             if output.len() == self.current_grid_size * self.current_grid_size {
                                  draw_grid(ui, output, self.current_grid_size, self.current_grid_size, 4.0);
                             } else {
                                 ui.label("(Invalid output state size)");
                             }
                        } else {
                            ui.label("Error: Invalid display iteration");
                        }

                        if let Some(total_iters) = self.iterations {
                            // Add slider to view iterations
                            let num_states = states.len();
                            let max_slider_idx = num_states.saturating_sub(1);
                            let mut current_slider_val = self.display_iteration.unwrap_or(0).min(max_slider_idx);

                            ui.add_space(10.0);
                            if ui.add(egui::Slider::new(&mut current_slider_val, 0..=max_slider_idx).text("View Iteration")).changed() {
                                self.display_iteration = Some(current_slider_val);
                            }

                            let label_text = if total_iters < self.max_iterations {
                                format!("Converged in {} iterations.", total_iters)
                            } else {
                                format!("Stopped after {} iterations.", total_iters)
                            };
                             ui.label(label_text);
                             ui.label(format!("Showing state at iteration: {}", iteration_to_display));
                        }
                    } else if self.network.is_none() {
                         ui.label("(Train network first)");
                    } else {
                        ui.label("(Run network to see output)");
                    }
                });
            });

            ui.separator();

            // Bottom part: Energy Plot
            ui.label("Energy Profile:");
            let plot_height = ui.available_height() * 0.8; // Use portion of remaining height
            if let Some(energies) = &self.energy_history {
                if !energies.is_empty() {
                    let points: PlotPoints = energies
                        .iter()
                        .enumerate()
                        .map(|(i, &e)| [i as f64, e])
                        .collect();
                    
                    let line = Line::new(points);
                    Plot::new("energy_plot")
                        .view_aspect(2.0)
                        .height(plot_height)
                        .show(ui, |plot_ui| {
                            plot_ui.line(line);
                        });
                } else {
                    ui.label("(No energy data)");
                }
            } else {
                ui.label("(Run network to calculate energy)");
            }

        });
    }
}

// Placeholder for the logic that should exist in lib.rs
// We'll replace run_and_print_ascii with a version that just returns the result.
/* // Remove the temporary run_network_logic
impl MyApp {
    fn run_network_logic(&self, network: &HopfieldNetwork) -> Result<(Vec<f64>, usize), HopfieldError> {
        // This is a temporary stand-in. Ideally, lib.rs would have a function like:
        // pub fn run(&self, initial_state: &[f64], max_iterations: usize) -> Result<(Vec<f64>, usize), HopfieldError>
        // For now, let's simulate it here based on update_step
        let mut current_state = self.input_state.clone();
        // Validate initial state (copied from run_and_print_ascii, needs proper validation access)
        if current_state.len() != network.size() {
             return Err(HopfieldError::DimensionMismatch("Initial state size mismatch".into()));
        }
         for &val in &current_state {
            if val != 1.0 && val != -1.0 {
                 return Err(HopfieldError::InvalidStateValue("Initial state value invalid".into()));
            }
        }


        for i in 0..self.max_iterations {
            let next_state = network.update_step(&current_state)?;
            if current_state == next_state {
                return Ok((current_state, i)); // Converged
            }
            current_state = next_state;
        }
        Ok((current_state, self.max_iterations)) // Max iterations reached
    }
}
*/