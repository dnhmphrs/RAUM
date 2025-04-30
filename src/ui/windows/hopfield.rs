use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use rand::rngs::ThreadRng;
use std::collections::HashSet;
use rusttype::{point, Font, Scale};

use crate::neural::hopfield::{HopfieldNetwork};
use crate::ui::widgets::grid::{draw_grid, apply_noise};
use crate::ui::windows::Window;

#[derive(Debug, PartialEq, Clone, Copy)]
enum UpdateMode {
    Synchronous,
    Asynchronous,
}

pub struct HopfieldWindow {
    // Neural network
    network: Option<HopfieldNetwork>,
    
    // Pattern data
    all_generated_patterns: Vec<(char, Vec<f64>)>,
    patterns: Vec<Vec<f64>>,
    trained_chars: Vec<char>,
    
    // UI State 
    current_grid_size: usize,
    available_chars: Vec<char>,
    selected_indices_for_training: HashSet<usize>,
    selected_pattern_index_for_input: Option<usize>,
    noise_level: f32,
    input_state: Vec<f64>,
    
    // Output state
    output_states: Option<Vec<Vec<f64>>>,
    energy_history: Option<Vec<f64>>,
    display_iteration: Option<usize>,
    iterations: Option<usize>,
    
    // Configuration
    error_message: Option<String>,
    max_iterations: usize,
    rng: ThreadRng,
    update_mode: UpdateMode,
}

impl HopfieldWindow {
    pub fn new() -> Self {
        let initial_grid_size = 16; // Smaller default for better performance
        let available_chars: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect();
        let initial_selected_indices: HashSet<usize> = (0..5).collect();
        
        // Generate initial set of all patterns (A-Z) at default size
        let all_generated_patterns = Self::get_patterns(initial_grid_size, &available_chars);

        // Create the initial training subset based on default selection
        let (patterns, trained_chars) = Self::filter_patterns(&all_generated_patterns, &initial_selected_indices);

        // Select the first pattern of the training subset as the initial input
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
            rng: rand::thread_rng(),
            update_mode: UpdateMode::Synchronous,
        }
    }
    
    // Helper function to filter the full pattern set based on selected indices
    fn filter_patterns(all_patterns: &[(char, Vec<f64>)], selected_indices: &HashSet<usize>) -> (Vec<Vec<f64>>, Vec<char>) {
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
    
    // Generate patterns of specified size for given characters using rusttype
    fn get_patterns(grid_size: usize, characters: &[char]) -> Vec<(char, Vec<f64>)> {
        // --- Configuration ---
        let font_path = "assets/font.otf"; // Ensure this font file exists at the project root
        let reference_pixel_height = 100.0; // Render large initially for bounds
        let threshold = 0.5; // Coverage threshold for 'on'
        let target_width = grid_size as f32;
        let target_height = grid_size as f32;
        let num_neurons_dynamic = grid_size * grid_size;
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
                // Fallback to basic patterns
                return Self::get_fallback_patterns(grid_size, characters);
            }
        };

        let font = match Font::try_from_vec(font_data) {
            Some(f) => f,
            None => {
                eprintln!(
                    "Error parsing font file '{}'. Using fallback patterns.",
                    font_path
                );
                return Self::get_fallback_patterns(grid_size, characters);
            }
        };

        let mut patterns = Vec::with_capacity(characters.len());

        for char_code in characters.iter() {
            let mut pattern = vec![-1.0; num_neurons_dynamic];

            let large_scale = Scale::uniform(reference_pixel_height);
            let large_glyph = font.glyph(*char_code).scaled(large_scale);
            let positioned_large_glyph = large_glyph.positioned(point(0.0, 0.0));

            if let Some(large_bb) = positioned_large_glyph.pixel_bounding_box() {
                let bb_width = large_bb.width() as f32;
                let bb_height = large_bb.height() as f32;

                if bb_width > 0.0 && bb_height > 0.0 {
                    let scale_x = target_width / bb_width;
                    let scale_y = target_height / bb_height;
                    let final_scale_factor = scale_x.min(scale_y);
                    let final_pixel_height = reference_pixel_height * final_scale_factor;
                    let final_scale = Scale::uniform(final_pixel_height);

                    let final_glyph = font.glyph(*char_code).scaled(final_scale);
                    let positioned_final_glyph = final_glyph.positioned(point(0.0, 0.0));

                    if let Some(final_bb) = positioned_final_glyph.pixel_bounding_box() {
                        let final_bb_width = final_bb.width() as f32;
                        let final_bb_height = final_bb.height() as f32;
                        let target_x = ((target_width - final_bb_width) / 2.0).round();
                        let target_y = ((target_height - final_bb_height) / 2.0).round();

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
                    }
                }
            }

            patterns.push((*char_code, pattern));
        }
        println!("Successfully generated {} patterns from font.", patterns.len());
        patterns
    }

    // Added fallback pattern generation logic
    fn get_fallback_patterns(grid_size: usize, characters: &[char]) -> Vec<(char, Vec<f64>)> {
        let num_neurons = grid_size * grid_size;
        let mut patterns = Vec::with_capacity(characters.len());
        
        for (i, &c) in characters.iter().enumerate() {
            let mut pattern = vec![-1.0; num_neurons];
            match i % 4 {
                0 => { // Horizontal line
                    for x in 0..grid_size { let y = grid_size / 3; pattern[y * grid_size + x] = 1.0; }
                },
                1 => { // Vertical line
                    for y in 0..grid_size { let x = grid_size / 2; pattern[y * grid_size + x] = 1.0; }
                },
                2 => { // Cross
                    for i in 0..grid_size { pattern[i * grid_size + i] = 1.0; pattern[i * grid_size + (grid_size - 1 - i)] = 1.0; }
                },
                3 => { // Box
                    for i in 0..grid_size { pattern[i] = 1.0; pattern[(grid_size - 1) * grid_size + i] = 1.0; pattern[i * grid_size] = 1.0; pattern[i * grid_size + (grid_size - 1)] = 1.0; }
                },
                _ => {}
            }
            patterns.push((c, pattern));
        }
        eprintln!("Generated {} fallback patterns.", patterns.len());
        patterns
    }
    
    // Updates the input state based on the selected pattern and noise level
    fn update_input_state(&mut self) {
        if let Some(idx) = self.selected_pattern_index_for_input {
            if let Some(pattern) = self.patterns.get(idx) {
                if pattern.len() == self.current_grid_size * self.current_grid_size {
                    self.input_state = apply_noise(pattern, self.noise_level);
                    // Reset output
                    self.output_states = None;
                    self.energy_history = None;
                    self.display_iteration = None;
                    self.iterations = None;
                }
            }
        }
    }
    
    // Handle grid size change
    fn handle_grid_size_change(&mut self, new_size: usize) {
        println!("Grid size changed to: {}", new_size);
        self.current_grid_size = new_size;
        
        // Regenerate all patterns at the new size
        self.all_generated_patterns = Self::get_patterns(self.current_grid_size, &self.available_chars);
        
        // Update the training subset
        let (new_patterns, new_trained_chars) = Self::filter_patterns(
            &self.all_generated_patterns, 
            &self.selected_indices_for_training
        );
        
        self.patterns = new_patterns;
        self.trained_chars = new_trained_chars;
        
        // Reset network and output
        self.network = None;
        self.output_states = None;
        self.energy_history = None;
        self.iterations = None;
        self.display_iteration = None;
        
        // Reset selected input pattern index if it's no longer valid
        self.selected_pattern_index_for_input = if self.patterns.is_empty() { 
            None 
        } else { 
            Some(0) 
        };
        
        // Update input state
        self.update_input_state();
    }
    
    // Train the network
    fn train_network(&mut self) {
        if self.patterns.is_empty() {
            self.error_message = Some("Cannot train: No patterns selected.".to_string());
            return;
        }
        
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
    
    // Run the network
    fn run_network(&mut self) {
        if self.input_state.len() != self.current_grid_size * self.current_grid_size {
            self.error_message = Some("Cannot run: Input state size mismatch.".to_string());
            return;
        }
        
        self.error_message = None;
        
        if let Some(net) = &self.network {
            // Check network size matches current grid size before running
            if net.size() != self.current_grid_size * self.current_grid_size {
                self.error_message = Some(
                    "Cannot run: Network size does not match current grid size. Retrain network.".to_string()
                );
                return;
            }
            
            // Call appropriate run method based on mode
            let run_result = match self.update_mode {
                UpdateMode::Synchronous => {
                    net.run(&self.input_state, self.max_iterations)
                }
                UpdateMode::Asynchronous => {
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
                    self.energy_history = None;
                    self.iterations = None;
                    self.error_message = Some(format!("Runtime Error: {}", e));
                }
            }
        }
    }
}

impl Window for HopfieldWindow {
    fn name(&self) -> &str {
        "Hopfield Network"
    }

    fn show_config(&mut self, ui: &mut egui::Ui) {
        // --- Controls Panel Content (Moved from SidePanel::left) ---
        ui.heading("Controls");
        ui.separator();
        
        // Grid Size
        ui.label("Grid Size (N x N):");
        let mut grid_size_mut = self.current_grid_size;
        let grid_slider = ui.add(egui::DragValue::new(&mut grid_size_mut)
            .speed(1.0)
            .range(8..=32) // Limit range for performance
            .clamp_to_range(true));

        if grid_slider.changed() {
            if grid_size_mut != self.current_grid_size {
                self.handle_grid_size_change(grid_size_mut);
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
                    .num_columns(5)
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
                            if col_count % 5 == 0 {
                                ui.end_row();
                            }
                        }
                    });
            });

        if selection_changed {
            println!("Training selection changed: {:?}", self.selected_indices_for_training);
            
            // Update the training subset
            let (new_patterns, new_trained_chars) = Self::filter_patterns(
                &self.all_generated_patterns, 
                &self.selected_indices_for_training
            );
            
            self.patterns = new_patterns;
            self.trained_chars = new_trained_chars;
            
            // Reset network and output
            self.network = None;
            self.output_states = None;
            self.energy_history = None;
            self.iterations = None;
            self.display_iteration = None;
            
            // Reset selected input pattern
            self.selected_pattern_index_for_input = if self.patterns.is_empty() { 
                None 
            } else { 
                Some(0) 
            };
            
            // Update input state
            self.update_input_state();
        }

        ui.separator();

        // Train Button
        if ui.button("Train Network").clicked() {
            self.train_network();
        }

        ui.separator();

        // --- Pattern Selection ---
        ui.vertical_centered(|ui| {
            ui.label("Select Initial Pattern (from trained set):");
        });
        
        egui::ScrollArea::vertical()
            .id_source("input_pattern_scroll")
            .max_height(150.0)
            .show(ui, |ui| {
                // Use a grid layout for better spacing
                egui::Grid::new("input_pattern_grid")
                    .num_columns(2)
                    .spacing([10.0, 5.0])
                    .show(ui, |ui| {
                        let mut clicked_index = None; // Variable to store the index if clicked
                        for (subset_idx, &char_code) in self.trained_chars.iter().enumerate() {
                            let label_text = char_code.to_string();
                            let is_selected = self.selected_pattern_index_for_input == Some(subset_idx);
                            
                            // Column 1: Selectable Label
                            if ui.selectable_label(is_selected, &label_text).clicked() {
                                // Store the index if the selection changed
                                if self.selected_pattern_index_for_input != Some(subset_idx) {
                                    clicked_index = Some(subset_idx);
                                }
                            }
                            
                            // Column 2: Preview Grid
                            if let Some(pattern) = self.patterns.get(subset_idx) {
                                if pattern.len() == self.current_grid_size * self.current_grid_size {
                                    draw_grid(ui, pattern, self.current_grid_size, self.current_grid_size, 2.0);
                                } else {
                                    ui.label("(Invalid preview size)");
                                }
                            } else {
                                ui.label("(No preview)");
                            }
                            ui.end_row();
                        }
                        
                        // Perform update after the loop if an index was clicked
                        if let Some(idx) = clicked_index {
                            self.selected_pattern_index_for_input = Some(idx);
                            self.update_input_state(); 
                        }
                    });
            });

        // --- Bottom Controls ---
        ui.separator();
        
        // Noise Control
        ui.label("Noise Level:");
        let noise_slider = ui.add(egui::Slider::new(&mut self.noise_level, 0.0..=1.0).text("Noise"));
        if noise_slider.changed() {
            self.update_input_state();
        }
        
        ui.separator();
        
        // Update Mode Selection
        ui.label("Update Mode:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.update_mode, UpdateMode::Synchronous, "Synchronous");
            ui.radio_value(&mut self.update_mode, UpdateMode::Asynchronous, "Asynchronous");
        });
        
        ui.separator();
        
        // Run Controls
        ui.label("Max Iterations:");
        ui.add(egui::DragValue::new(&mut self.max_iterations).speed(1.0).range(1..=1000));
        
        ui.separator();
        
        if ui.add_enabled(self.network.is_some(), egui::Button::new("Run Network")).clicked() {
            self.run_network();
        }
        
        // --- Info Section ---
        ui.separator();
        
        egui::CollapsingHeader::new("Info & Formulae")
            .id_source("info_collapse")
            .show(ui, |ui| {
                ui.label(format!("Grid Size: {}x{}", self.current_grid_size, self.current_grid_size));
                ui.label(format!("Neurons: {}", self.current_grid_size * self.current_grid_size));
                ui.label(format!("Stored Patterns: {}", self.patterns.len()));
                ui.separator();
                ui.label("Update Rule: Sᵢ(t+1) = sgn( Σⱼ Wᵢⱼ Sⱼ(t) )");
                ui.label("Learning Rule: Wᵢⱼ = Σₚ ξᵢᵖ ξⱼᵖ  (i ≠ j, Wᵢᵢ = 0)");
                ui.label("Energy: E = -½ Σᵢ Σⱼ Wᵢⱼ Sᵢ Sⱼ (i ≠ j)");
            });
            
        // Display Error Messages
        if let Some(err) = &self.error_message {
            ui.separator();
            ui.colored_label(egui::Color32::RED, err);
        }
    }
    
    fn show_content(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        // --- Content Panel Content (Moved from CentralPanel::default) ---
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
                        if pattern.len() == self.current_grid_size * self.current_grid_size {
                            draw_grid(ui, pattern, self.current_grid_size, self.current_grid_size, 4.0); 
                            // Show character label
                            let label_text = self.trained_chars.get(idx)
                                .map(|&c| c.to_string())
                                .unwrap_or_else(|| "?".to_string());
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
                if self.input_state.len() == self.current_grid_size * self.current_grid_size {
                    draw_grid(ui, &self.input_state, self.current_grid_size, self.current_grid_size, 4.0); 
                } else {
                    ui.label("(Invalid input state size)");
                }
                ui.label(format!("Noise: {:.2}", self.noise_level));
            });

            // Column 3: Output State (Iteration Viewer)
            columns[2].vertical_centered(|ui|{
                ui.label("Output State");
                ui.separator();
                if let Some(states) = &self.output_states {
                    // Get the state to display based on the slider
                    let iteration_to_display = self.display_iteration
                        .unwrap_or(0)
                        .min(states.len().saturating_sub(1));
                        
                    if let Some(output) = states.get(iteration_to_display) {
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
                        let mut current_slider_val = self.display_iteration
                            .unwrap_or(0)
                            .min(max_slider_idx);

                        ui.add_space(10.0);
                        if ui.add(egui::Slider::new(&mut current_slider_val, 0..=max_slider_idx)
                            .text("Iteration"))
                            .changed() 
                        {
                            self.display_iteration = Some(current_slider_val);
                        }

                        let label_text = if total_iters < self.max_iterations {
                            format!("Converged in {} iterations.", total_iters)
                        } else {
                            format!("Stopped after {} iterations.", total_iters)
                        };
                        ui.label(label_text);
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
        let plot_height = ui.available_height() * 0.8;
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
    }
}
