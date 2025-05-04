use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use rand::rngs::ThreadRng;
use std::collections::HashSet;
use rusttype::{point, Font, Scale};

use crate::neural::hopfield::{HopfieldNetwork, TrainingRule};
use crate::ui::widgets::grid::{draw_grid, apply_noise};
use crate::ui::windows::Window;

#[derive(Debug, PartialEq, Clone, Copy)]
enum GraphType {
    FullyConnected,
    ErdosRenyi,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum UpdateMode {
    Synchronous,
    Asynchronous,
}

pub struct HopfieldWindow {
    // Neural network
    network: Option<HopfieldNetwork>,
    
    // Pattern data
    all_generated_patterns: Vec<(char, Vec<f64>)>, // (Char, Original Pattern from font)
    patterns: Vec<Vec<f64>>, // Currently ACTIVE patterns for training/input selection
    trained_chars: Vec<char>, // Chars corresponding to `patterns`
    
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
    beta: f64, 
    pattern_overlap: Option<Vec<Vec<f64>>>, 
    training_rule: TrainingRule,
    overlap_histogram: Option<Vec<egui_plot::Bar>>,
    graph_type: GraphType,
    er_connectivity: f64,
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
            patterns: patterns.clone(),
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
            beta: 1.0,
            pattern_overlap: Self::calculate_overlap_matrix(&patterns),
            training_rule: TrainingRule::PseudoInverse,
            overlap_histogram: Self::calculate_overlap_histogram(&Self::calculate_overlap_matrix(&patterns)),
            graph_type: GraphType::FullyConnected,
            er_connectivity: 1.0,
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
        self.pattern_overlap = Self::calculate_overlap_matrix(&self.patterns);
        self.overlap_histogram = Self::calculate_overlap_histogram(&self.pattern_overlap);
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
                    net.run(&self.input_state, self.max_iterations, self.beta, &mut self.rng)
                }
                UpdateMode::Asynchronous => {
                    net.run_async(&self.input_state, self.max_iterations, self.beta, &mut self.rng)
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

    // Helper function to calculate histogram data for off-diagonal overlaps
    fn calculate_overlap_histogram(overlap_matrix: &Option<Vec<Vec<f64>>>) -> Option<Vec<egui_plot::Bar>> {
        let matrix = overlap_matrix.as_ref()?; // Return None if overlap_matrix is None

        if matrix.is_empty() {
            return None;
        }

        let num_patterns = matrix.len();
        let mut off_diagonal_overlaps = Vec::new();

        for i in 0..num_patterns {
            for j in (i + 1)..num_patterns { // Only upper triangle (excluding diagonal)
                // We care about the magnitude of correlation
                off_diagonal_overlaps.push(matrix[i][j].abs()); 
            }
        }

        if off_diagonal_overlaps.is_empty() {
            return Some(Vec::new()); // No off-diagonal elements (e.g., only 1 pattern)
        }

        // Define histogram bins (e.g., 10 bins from 0.0 to 1.0)
        let num_bins = 10;
        let bin_width = 1.0 / num_bins as f64;
        let mut bins = vec![0; num_bins];

        for &overlap in &off_diagonal_overlaps {
            let bin_index = (overlap / bin_width).floor() as usize;
            // Clamp index to handle overlap == 1.0 case
            let clamped_index = bin_index.min(num_bins - 1);
            bins[clamped_index] += 1;
        }

        // Create egui_plot Bars
        let bars: Vec<egui_plot::Bar> = bins.into_iter().enumerate().map(|(index, count)| {
            let x = (index as f64 + 0.5) * bin_width; // Center of the bin
            egui_plot::Bar::new(x, count as f64).width(bin_width * 0.9) // Make bars slightly thinner than bin
        }).collect();

        Some(bars)
    }

    // Helper function to calculate the overlap matrix between patterns
    fn calculate_overlap_matrix(patterns: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        if patterns.is_empty() || patterns[0].is_empty() {
            return None;
        }

        let num_patterns = patterns.len();
        let num_neurons = patterns[0].len() as f64;
        let mut overlap_matrix = vec![vec![0.0; num_patterns]; num_patterns];

        for p in 0..num_patterns {
            for q in p..num_patterns { // Calculate only upper triangle + diagonal
                let mut dot_product = 0.0;
                for i in 0..patterns[p].len() {
                    dot_product += patterns[p][i] * patterns[q][i];
                }
                let overlap = dot_product / num_neurons;
                overlap_matrix[p][q] = overlap;
                if p != q { // Mirror to lower triangle
                    overlap_matrix[q][p] = overlap;
                }
            }
        }
        Some(overlap_matrix)
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
            self.pattern_overlap = Self::calculate_overlap_matrix(&self.patterns);
            self.overlap_histogram = Self::calculate_overlap_histogram(&self.pattern_overlap);
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

        // --- Training Rule Selection ---
        ui.label("Training Rule:");
        ui.horizontal(|ui| {
            let changed = ui.radio_value(&mut self.training_rule, TrainingRule::Hebbian, "Hebbian").changed();
            let changed = changed || ui.radio_value(&mut self.training_rule, TrainingRule::PseudoInverse, "Pseudo-Inverse").changed();
            if changed {
                self.network = None; // Require retraining if rule changes
                println!("Training rule changed to {:?}. Retrain network.", self.training_rule);
            }
        });

        ui.separator();

        // --- Graph Topology Selection ---
        ui.label("Graph Topology:");
        let mut topology_changed = false;
        ui.horizontal(|ui| {
            topology_changed |= ui.radio_value(&mut self.graph_type, GraphType::FullyConnected, "Fully Connected").changed();
            topology_changed |= ui.radio_value(&mut self.graph_type, GraphType::ErdosRenyi, "Erdős-Rényi").changed();
        });

        // Only show connectivity slider if Erdős-Rényi is selected
        if self.graph_type == GraphType::ErdosRenyi {
            ui.add_space(5.0);
            ui.label("ER Connectivity (p):");
            let er_slider = ui.add(egui::Slider::new(&mut self.er_connectivity, 0.0..=1.0).text("Probability"));
            if er_slider.changed() {
                topology_changed = true;
            }
        }

        if topology_changed {
            self.network = None; // Require retraining if topology settings change
            println!("Graph topology settings changed. Retrain network.");
        }

        ui.separator();

        // Train Button
        if ui.button("Train Network").clicked() {
            // Logic previously in train_network method
            if self.patterns.is_empty() {
                self.error_message = Some("Cannot train: No patterns selected.".to_string());
            } else {
                self.error_message = None;
                self.output_states = None;
                self.energy_history = None;
                self.iterations = None;
                self.display_iteration = None;

                // Create network first
                let mut net = HopfieldNetwork::new(self.current_grid_size * self.current_grid_size); 

                // Train using the selected rule
                match net.train(&self.patterns, self.training_rule) { 
                    Ok(_) => {
                        // Apply topology modification if necessary
                        if self.graph_type == GraphType::ErdosRenyi {
                            net.apply_erdos_renyi_topology(self.er_connectivity, &mut self.rng);
                        }
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
                            // Use self.patterns which holds the currently active set (original or ortho)
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

        ui.separator();

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
        
        // Beta Control
        ui.label("Update Rule Beta (Temperature Inverse):");
        ui.add(egui::DragValue::new(&mut self.beta).speed(0.01).range(0.01..=10.0));
        
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
                ui.separator();
                ui.label("Pattern Overlap Matrix (m = 1/N * ξᵖ⋅ξ۹):");

                // Display Overlap Matrix
                if let Some(matrix) = &self.pattern_overlap {
                    if !self.trained_chars.is_empty() && !matrix.is_empty() {
                        egui::Grid::new("overlap_matrix_grid")
                            .num_columns(self.trained_chars.len() + 1)
                            .min_col_width(30.0)
                            .spacing([5.0, 5.0])
                            .show(ui, |ui| {
                                // Header Row (Characters)
                                ui.label(""); // Top-left corner
                                for &char_code in &self.trained_chars {
                                    ui.label(char_code.to_string()).rect.width();
                                }
                                ui.end_row();

                                // Matrix Rows
                                for (p, &char_code) in self.trained_chars.iter().enumerate() {
                                    ui.label(char_code.to_string()); // Row Header
                                    for q in 0..self.trained_chars.len() {
                                        let overlap = matrix[p][q];
                                        // Color based on overlap value (closer to +/-1 is less ideal)
                                        let abs_overlap = overlap.abs();
                                        let color = if p == q { // Diagonal
                                            egui::Color32::WHITE
                                        } else if abs_overlap > 0.7 {
                                            egui::Color32::from_rgb(255, 100, 100) // Reddish for high overlap
                                        } else if abs_overlap > 0.3 {
                                            egui::Color32::from_rgb(255, 200, 100) // Orange/Yellow for medium
                                        } else {
                                            egui::Color32::from_rgb(100, 255, 100) // Greenish for low
                                        };
                                        ui.colored_label(color, format!("{:.2}", overlap));
                                    }
                                    ui.end_row();
                                }
                            });
                    } else {
                        ui.label("(No patterns selected for overlap calculation)");
                    }
                } else {
                    ui.label("(Overlap not calculated)");
                }

                ui.separator();
                ui.label("Histogram of Off-Diagonal Overlap Magnitudes (|m_pq|, p != q):");

                // Display Overlap Histogram
                if let Some(histogram_bars) = &self.overlap_histogram {
                    if !histogram_bars.is_empty() {
                        let chart = egui_plot::BarChart::new(histogram_bars.clone()) 
                            .color(egui::Color32::LIGHT_BLUE)
                            .name("Overlap Distribution");

                        egui_plot::Plot::new("overlap_histogram_plot")
                            .legend(egui_plot::Legend::default())
                            .height(100.0) // Adjust height as needed
                            .show_axes([true, true])
                            .show(ui, |plot_ui| {
                                plot_ui.bar_chart(chart);
                            });
                    } else {
                        ui.label("(Not enough patterns for histogram)");
                    }
                } else {
                    ui.label("(Histogram not calculated)");
                }
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
