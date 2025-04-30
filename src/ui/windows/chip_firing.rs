use eframe::egui;
use egui_plot::{Plot, PlotPoints, Points};
use rand::rngs::ThreadRng;

use crate::neural::chip_firing::{ChipFiringGraph, UpdateMode, VertexSelectionStrategy};
use crate::ui::windows::Window;

/// Predefined graph types for the UI
#[derive(Debug, Clone, Copy, PartialEq)]
enum GraphType {
    Grid,
    Cycle,
    Complete,
    Star,
    Custom,
}

/// Visualization mode for the chip firing graph
#[derive(Debug, Clone, Copy, PartialEq)]
enum VisualizationMode {
    /// Display the graph as a network
    Network,
    /// Display the graph as a grid (only for grid graphs)
    Grid,
    /// Display the chip distribution as a bar chart
    BarChart,
}

/// Window for chip firing graph simulation and visualization
pub struct ChipFiringWindow {
    /// The chip firing graph model
    graph: Option<ChipFiringGraph>,
    
    /// Configuration for graph creation
    graph_type: GraphType,
    graph_size: usize,        // For cycle and complete graphs
    grid_width: usize,        // For grid graphs
    grid_height: usize,       // For grid graphs
    custom_edges: String,     // For custom graphs, format: "0,1 1,2 ..."
    
    /// Simulation parameters
    max_steps: usize,
    step_interval: f64,       // In seconds
    last_step_time: f64,
    auto_step: bool,
    
    /// Display settings
    display_step: usize,
    visualization_mode: VisualizationMode,
    show_active_vertices: bool,
    vertex_radius: f32,
    edge_thickness: f32,
    grid_cell_size: f32,
    
    /// Vertex interaction
    selected_vertex: Option<usize>,
    add_chip_to_selected: bool,
    
    /// Node positions for network visualization
    node_positions: Vec<egui::Vec2>,
    
    /// Random number generator for vertex selection
    rng: ThreadRng,
    
    /// UI state
    error_message: Option<String>,
}

impl ChipFiringWindow {
    pub fn new() -> Self {
        Self {
            graph: None,
            graph_type: GraphType::Grid,
            graph_size: 10,
            grid_width: 5,
            grid_height: 5,
            custom_edges: String::new(),
            max_steps: 100,
            step_interval: 0.2,
            last_step_time: 0.0,
            auto_step: false,
            display_step: 0,
            visualization_mode: VisualizationMode::Network,
            show_active_vertices: true,
            vertex_radius: 15.0,
            edge_thickness: 2.0,
            grid_cell_size: 50.0,
            selected_vertex: None,
            add_chip_to_selected: false,
            node_positions: Vec::new(),
            rng: rand::thread_rng(),
            error_message: None,
        }
    }
    
    /// Create a new chip firing graph based on current settings
    fn create_graph(&mut self) -> Result<ChipFiringGraph, String> {
        match self.graph_type {
            GraphType::Grid => {
                // Create a grid graph
                if self.grid_width == 0 || self.grid_height == 0 {
                    return Err("Grid dimensions must be greater than 0".to_string());
                }
                
                let num_vertices = self.grid_width * self.grid_height;
                let initial_config = vec![0; num_vertices];
                
                ChipFiringGraph::new_grid(self.grid_width, self.grid_height, initial_config)
                    .map_err(|e| format!("Failed to create grid graph: {}", e))
            },
            GraphType::Cycle => {
                // Create a cycle graph
                if self.graph_size < 3 {
                    return Err("Cycle graph needs at least 3 vertices".to_string());
                }
                
                let mut edges = Vec::new();
                for i in 0..self.graph_size {
                    edges.push((i, (i + 1) % self.graph_size));
                }
                
                let initial_config = vec![0; self.graph_size];
                
                ChipFiringGraph::from_edge_list(&edges, self.graph_size, initial_config)
                    .map_err(|e| format!("Failed to create cycle graph: {}", e))
            },
            GraphType::Complete => {
                // Create a complete graph (all vertices connected to all others)
                if self.graph_size < 2 {
                    return Err("Complete graph needs at least 2 vertices".to_string());
                }
                
                let mut edges = Vec::new();
                for i in 0..self.graph_size {
                    for j in (i+1)..self.graph_size {
                        edges.push((i, j));
                    }
                }
                
                let initial_config = vec![0; self.graph_size];
                
                ChipFiringGraph::from_edge_list(&edges, self.graph_size, initial_config)
                    .map_err(|e| format!("Failed to create complete graph: {}", e))
            },
            GraphType::Star => {
                // Create a star graph (center connected to all others)
                if self.graph_size < 3 {
                    return Err("Star graph needs at least 3 vertices".to_string());
                }
                
                let mut edges = Vec::new();
                for i in 1..self.graph_size {
                    edges.push((0, i)); // Connect center (0) to all others
                }
                
                let initial_config = vec![0; self.graph_size];
                
                ChipFiringGraph::from_edge_list(&edges, self.graph_size, initial_config)
                    .map_err(|e| format!("Failed to create star graph: {}", e))
            },
            GraphType::Custom => {
                // Parse custom edges from string
                let edges_result = self.parse_custom_edges();
                match edges_result {
                    Ok((edges, num_vertices)) => {
                        let initial_config = vec![0; num_vertices];
                        
                        ChipFiringGraph::from_edge_list(&edges, num_vertices, initial_config)
                            .map_err(|e| format!("Failed to create custom graph: {}", e))
                    },
                    Err(e) => Err(e),
                }
            },
        }
    }
    
    /// Parse custom edges string into edge list and vertex count
    fn parse_custom_edges(&self) -> Result<(Vec<(usize, usize)>, usize), String> {
        let mut edges = Vec::new();
        let mut max_vertex = 0;
        
        for edge_str in self.custom_edges.split_whitespace() {
            let parts: Vec<&str> = edge_str.split(',').collect();
            if parts.len() != 2 {
                return Err(format!("Invalid edge format: '{}'. Use 'from,to' format.", edge_str));
            }
            
            let from = parts[0].parse::<usize>().map_err(|_| {
                format!("Invalid vertex index: '{}' in edge '{}'", parts[0], edge_str)
            })?;
            
            let to = parts[1].parse::<usize>().map_err(|_| {
                format!("Invalid vertex index: '{}' in edge '{}'", parts[1], edge_str)
            })?;
            
            edges.push((from, to));
            max_vertex = max_vertex.max(from).max(to);
        }
        
        // Number of vertices is max index + 1
        let num_vertices = max_vertex + 1;
        
        if edges.is_empty() {
            return Err("No valid edges provided".to_string());
        }
        
        Ok((edges, num_vertices))
    }
    
    /// Calculate node positions for network visualization
    fn calculate_node_positions(&mut self) {
        if let Some(graph) = &self.graph {
            let n = graph.num_vertices;
            self.node_positions.clear();
            
            match self.graph_type {
                GraphType::Grid => {
                    // Position nodes in a grid layout
                    for y in 0..self.grid_height {
                        for x in 0..self.grid_width {
                            let pos = egui::Vec2::new(
                                x as f32 * 100.0,
                                y as f32 * 100.0,
                            );
                            self.node_positions.push(pos);
                        }
                    }
                },
                GraphType::Cycle => {
                    // Position nodes in a circle
                    let radius = 200.0;
                    let center = egui::Vec2::new(250.0, 250.0);
                    
                    for i in 0..n {
                        let angle = (i as f32 / n as f32) * 2.0 * std::f32::consts::PI;
                        let pos = egui::Vec2::new(
                            center.x + radius * angle.cos(),
                            center.y + radius * angle.sin(),
                        );
                        self.node_positions.push(pos);
                    }
                },
                GraphType::Complete | GraphType::Star | GraphType::Custom => {
                    // Position nodes in a circle for these graph types too
                    let radius = 200.0;
                    let center = egui::Vec2::new(250.0, 250.0);
                    
                    // For Star, place center at the middle
                    if self.graph_type == GraphType::Star {
                        self.node_positions.push(center);
                        for i in 1..n {
                            let angle = ((i-1) as f32 / (n-1) as f32) * 2.0 * std::f32::consts::PI;
                            let pos = egui::Vec2::new(
                                center.x + radius * angle.cos(),
                                center.y + radius * angle.sin(),
                            );
                            self.node_positions.push(pos);
                        }
                    } else {
                        // Circle layout for Complete and Custom
                        for i in 0..n {
                            let angle = (i as f32 / n as f32) * 2.0 * std::f32::consts::PI;
                            let pos = egui::Vec2::new(
                                center.x + radius * angle.cos(),
                                center.y + radius * angle.sin(),
                            );
                            self.node_positions.push(pos);
                        }
                    }
                },
            }
        }
    }
    
    /// Get the configuration at the current display step
    fn current_configuration(&self) -> Option<&Vec<i32>> {
        if let Some(graph) = &self.graph {
            if !graph.history.is_empty() {
                let step = self.display_step.min(graph.history.len() - 1);
                return Some(&graph.history[step]);
            }
        }
        None
    }
    
    /// Get active vertices at the current display step
    fn current_active_vertices(&self) -> Vec<usize> {
        if let Some(config) = self.current_configuration() {
            if let Some(graph) = &self.graph {
                // Identify which vertices would be active with this configuration
                let mut active = Vec::new();
                for i in 0..graph.num_vertices {
                    if config[i] >= graph.degrees[i] as i32 {
                        active.push(i);
                    }
                }
                return active;
            }
        }
        Vec::new()
    }
    
    /// Reset the graph to its initial configuration
    fn reset_graph(&mut self) {
        if let Some(graph) = &mut self.graph {
            graph.reset();
            self.display_step = 0;
        }
    }
    
    /// Execute a single step of the simulation
    fn step_simulation(&mut self) {
        if let Some(graph) = &mut self.graph {
            if let Err(e) = graph.step(&mut self.rng) {
                self.error_message = Some(format!("Simulation error: {}", e));
            } else {
                self.display_step = graph.history.len() - 1;
            }
        }
    }
    
    /// Initialize a random configuration
    fn randomize_configuration(&mut self) {
        if let Some(graph) = &mut self.graph {
            let _rng = rand::thread_rng();
            let mut new_config = Vec::with_capacity(graph.num_vertices);
            
            for i in 0..graph.num_vertices {
                // Random number of chips from 0 to degree
                let degree = graph.degrees[i] as i32;
                let chips = if rand::random::<bool>() {
                    rand::random::<i32>() % (degree + 1) 
                } else {
                    degree // Exactly the degree (active)
                };
                new_config.push(chips.max(0)); // Ensure non-negative
            }
            
            if let Err(e) = graph.set_configuration(new_config) {
                self.error_message = Some(format!("Failed to set random configuration: {}", e));
            } else {
                self.display_step = 0;
            }
        }
    }
    
    /// Add a chip to the selected vertex
    fn add_chip(&mut self) {
        if let (Some(graph), Some(vertex)) = (&mut self.graph, self.selected_vertex) {
            if vertex < graph.num_vertices {
                let mut new_config = graph.configuration.clone();
                new_config[vertex] += 1;
                
                if let Err(e) = graph.set_configuration(new_config) {
                    self.error_message = Some(format!("Failed to add chip: {}", e));
                } else {
                    self.display_step = 0;
                }
            }
        }
    }
    
    /// Remove a chip from the selected vertex
    fn remove_chip(&mut self) {
        if let (Some(graph), Some(vertex)) = (&mut self.graph, self.selected_vertex) {
            if vertex < graph.num_vertices && graph.configuration[vertex] > 0 {
                let mut new_config = graph.configuration.clone();
                new_config[vertex] -= 1;
                
                if let Err(e) = graph.set_configuration(new_config) {
                    self.error_message = Some(format!("Failed to remove chip: {}", e));
                } else {
                    self.display_step = 0;
                }
            }
        }
    }
    
    /// Trigger an avalanche at the selected vertex
    fn trigger_avalanche(&mut self) {
        if let (Some(graph), Some(vertex)) = (&mut self.graph, self.selected_vertex) {
            if vertex < graph.num_vertices {
                match graph.trigger_avalanche(vertex, self.max_steps, &mut self.rng) {
                    Ok(steps) => {
                        self.display_step = graph.history.len() - 1;
                        println!("Avalanche completed in {} steps", steps);
                    },
                    Err(e) => {
                        self.error_message = Some(format!("Avalanche error: {}", e));
                    }
                }
            }
        }
    }
    
    /// Draw the graph as a network (immutable self, takes painter)
    fn draw_network(&self, painter: &egui::Painter, response: &egui::Response) {
        if let Some(graph) = &self.graph {
            if self.node_positions.len() != graph.num_vertices {
                // Cannot draw if positions mismatch
                painter.text(
                    response.rect.center(), 
                    egui::Align2::CENTER_CENTER, 
                    "Error: Node positions mismatch", 
                    egui::FontId::default(), 
                    egui::Color32::RED
                );
                return;
            }
            
            // Get configuration and active vertices for current display step
            let config = self.current_configuration().unwrap_or(&graph.configuration);
            let active_vertices = if self.show_active_vertices {
                self.current_active_vertices()
            } else {
                Vec::new()
            };
            
            // Draw edges first
            for i in 0..graph.num_vertices {
                for &j in &graph.neighbors(i) {
                    if i < j { 
                        let start = self.node_positions[i];
                        let end = self.node_positions[j];
                        
                        painter.line_segment(
                            [response.rect.min + start, response.rect.min + end],
                            egui::Stroke::new(self.edge_thickness, egui::Color32::GRAY),
                        );
                    }
                }
            }
            
            // Draw nodes
            for i in 0..graph.num_vertices {
                let pos = response.rect.min + self.node_positions[i];
                let is_active = active_vertices.contains(&i);
                let is_selected = Some(i) == self.selected_vertex;
                
                let fill_color = if is_selected {
                    egui::Color32::YELLOW
                } else if is_active {
                    egui::Color32::GREEN
                } else {
                    egui::Color32::WHITE
                };
                
                painter.circle_filled(pos, self.vertex_radius, fill_color);
                painter.circle_stroke(pos, self.vertex_radius, egui::Stroke::new(2.0, egui::Color32::BLACK));
                
                let chip_count = config[i].to_string();
                painter.text(
                    pos,
                    egui::Align2::CENTER_CENTER,
                    chip_count,
                    egui::FontId::proportional(14.0),
                    egui::Color32::BLACK,
                );
            }
            
            // Interaction is handled by the caller (show_content)
        }
    }
    
    /// Draw the graph as a grid (immutable self, takes painter)
    fn draw_grid(&self, painter: &egui::Painter, response: &egui::Response) {
        if let Some(graph) = &self.graph {
            if self.graph_type != GraphType::Grid {
                 painter.text(
                    response.rect.center(), 
                    egui::Align2::CENTER_CENTER, 
                    "Grid view only for Grid graphs", 
                    egui::FontId::default(), 
                    egui::Color32::RED
                );
                return;
            }
            
            let config = self.current_configuration().unwrap_or(&graph.configuration);
            let active_vertices = if self.show_active_vertices {
                self.current_active_vertices()
            } else {
                Vec::new()
            };
            
            let grid_width = self.grid_width;
            let grid_height = self.grid_height;
            
            // Draw grid cells
            for y in 0..grid_height {
                for x in 0..grid_width {
                    let idx = y * grid_width + x;
                    let is_active = active_vertices.contains(&idx);
                    let is_selected = Some(idx) == self.selected_vertex;
                    
                    let fill_color = if is_selected {
                        egui::Color32::YELLOW
                    } else if is_active {
                        egui::Color32::GREEN
                    } else {
                        egui::Color32::WHITE
                    };
                    
                    let cell_pos = egui::Vec2::new(x as f32 * self.grid_cell_size, y as f32 * self.grid_cell_size);
                    let cell_rect = egui::Rect::from_min_size(
                        response.rect.min + cell_pos,
                        egui::vec2(self.grid_cell_size, self.grid_cell_size),
                    );
                    
                    painter.rect_filled(cell_rect, 0.0, fill_color);
                    painter.rect_stroke(cell_rect, 0.0, egui::Stroke::new(1.0, egui::Color32::BLACK));
                    
                    let chip_count = config[idx].to_string();
                    painter.text(
                        cell_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        chip_count,
                        egui::FontId::proportional(14.0),
                        egui::Color32::BLACK,
                    );
                }
            }
            
            // Interaction is handled by the caller (show_content)
        }
    }
    
    /// Draw chip distribution as a bar chart
    fn draw_bar_chart(&self, ui: &mut egui::Ui) {
        if let Some(graph) = &self.graph {
            if let Some(config) = self.current_configuration() {
                let max_chips = config.iter().cloned().max().unwrap_or(0);
                
                let chart = Plot::new("chip_distribution")
                    .height(300.0)
                    .y_axis_label("Chips")
                    .x_axis_label("Vertex")
                    .allow_scroll(false)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .y_axis_min_width(2.0)
                    .show_x(true)
                    .show_y(true)
                    .show_axes([true, true])
                    .include_y(0.0)
                    .include_y(max_chips as f64 + 1.0);
                
                chart.show(ui, |plot_ui| {
                    // Create bar chart data
                    let bar_width = 0.5;
                    
                    for (i, &chips) in config.iter().enumerate() {
                        // Each bar is made of 4 points forming a rectangle
                        let bar_points = PlotPoints::new(vec![
                            [(i as f64) - bar_width / 2.0, 0.0],
                            [(i as f64) - bar_width / 2.0, chips as f64],
                            [(i as f64) + bar_width / 2.0, chips as f64],
                            [(i as f64) + bar_width / 2.0, 0.0],
                        ]);
                        
                        // Color bars based on vertex status
                        let is_active = self.current_active_vertices().contains(&i);
                        let is_selected = Some(i) == self.selected_vertex;
                        
                        let color = if is_selected {
                            egui::Color32::YELLOW
                        } else if is_active {
                            egui::Color32::GREEN
                        } else {
                            egui::Color32::BLUE
                        };
                        
                        // Add bar as filled polygon
                        plot_ui.polygon(egui_plot::Polygon::new(bar_points).fill_color(color));
                    }
                    
                    // Overlay vertex degrees as points
                    let degrees: PlotPoints = graph.degrees
                        .iter()
                        .enumerate()
                        .map(|(i, &degree)| [i as f64, degree as f64])
                        .collect();
                    
                    plot_ui.points(Points::new(degrees)
                        .color(egui::Color32::RED)
                        .shape(egui_plot::MarkerShape::Circle)
                        .radius(5.0)
                        .name("Degrees (activation threshold)"));
                });
                
                // Add a note explaining what the chart shows
                ui.label("Blue bars: Chip count | Red dots: Degree (activation threshold) | Green: Active");
            }
        }
    }
}

impl Window for ChipFiringWindow {
    fn name(&self) -> &str {
        "Chip Firing Graph"
    }
    
    fn show_config(&mut self, ui: &mut egui::Ui) {
        // Graph creation settings
        ui.heading("Graph Settings");
        ui.separator();
        
        ui.horizontal(|ui| {
            ui.label("Graph Type:");
            ui.radio_value(&mut self.graph_type, GraphType::Grid, "Grid");
            ui.radio_value(&mut self.graph_type, GraphType::Cycle, "Cycle");
            ui.radio_value(&mut self.graph_type, GraphType::Complete, "Complete");
            ui.radio_value(&mut self.graph_type, GraphType::Star, "Star");
            ui.radio_value(&mut self.graph_type, GraphType::Custom, "Custom");
        });
        
        // Type-specific settings
        match self.graph_type {
            GraphType::Grid => {
                ui.horizontal(|ui| {
                    ui.label("Width:");
                    ui.add(egui::DragValue::new(&mut self.grid_width).speed(1.0).range(2..=20));
                    ui.label("Height:");
                    ui.add(egui::DragValue::new(&mut self.grid_height).speed(1.0).range(2..=20));
                });
            },
            GraphType::Cycle | GraphType::Complete | GraphType::Star => {
                ui.horizontal(|ui| {
                    ui.label("Number of Vertices:");
                    ui.add(egui::DragValue::new(&mut self.graph_size).speed(1.0).range(3..=50));
                });
            },
            GraphType::Custom => {
                ui.label("Enter edges as space-separated pairs (e.g., \"0,1 1,2 2,0\"):");
                ui.text_edit_multiline(&mut self.custom_edges);
            },
        }
        
        if ui.button("Create Graph").clicked() {
            match self.create_graph() {
                Ok(graph) => {
                    self.graph = Some(graph);
                    self.calculate_node_positions();
                    self.display_step = 0;
                    self.error_message = None;
                },
                Err(e) => {
                    self.error_message = Some(e);
                },
            }
        }
        
        ui.separator();
        
        // Simulation settings (only show if graph exists)
        if self.graph.is_some() {
            ui.heading("Simulation Settings");
            ui.separator();
            
            ui.horizontal(|ui| {
                ui.label("Update Mode:");
                if let Some(graph) = &mut self.graph {
                    ui.radio_value(&mut graph.update_mode, UpdateMode::Sequential, "Sequential");
                    ui.radio_value(&mut graph.update_mode, UpdateMode::Parallel, "Parallel");
                }
            });
            
            // Selection strategy (only for Sequential mode)
            if let Some(graph) = &mut self.graph {
                if graph.update_mode == UpdateMode::Sequential {
                    ui.horizontal(|ui| {
                        ui.label("Selection Strategy:");
                        ui.radio_value(&mut graph.selection_strategy, VertexSelectionStrategy::FirstActive, "First Active");
                        ui.radio_value(&mut graph.selection_strategy, VertexSelectionStrategy::RandomActive, "Random Active");
                    });
                }
            }
            
            ui.horizontal(|ui| {
                ui.label("Max Steps:");
                ui.add(egui::DragValue::new(&mut self.max_steps).speed(1.0).range(1..=1000));
            });
            
            ui.horizontal(|ui| {
                ui.label("Auto Step Interval:"); 
                ui.add(egui::DragValue::new(&mut self.step_interval).speed(0.1).range(0.1..=5.0));
                ui.label("seconds");
            });
            
            ui.checkbox(&mut self.add_chip_to_selected, "Add Chip on Click");
            
            ui.separator();
            
            // Visualization settings
            ui.heading("Visualization Settings");
            ui.separator();
            
            ui.horizontal(|ui| {
                ui.label("View Mode:");
                ui.radio_value(&mut self.visualization_mode, VisualizationMode::Network, "Network");
                
                // Only allow Grid mode for grid graphs
                if self.graph_type == GraphType::Grid {
                    ui.radio_value(&mut self.visualization_mode, VisualizationMode::Grid, "Grid");
                }
                
                ui.radio_value(&mut self.visualization_mode, VisualizationMode::BarChart, "Bar Chart");
            });
            
            ui.checkbox(&mut self.show_active_vertices, "Highlight Active Vertices");
            
            match self.visualization_mode {
                VisualizationMode::Network => {
                    ui.horizontal(|ui| {
                        ui.label("Vertex Radius:");
                        ui.add(egui::Slider::new(&mut self.vertex_radius, 5.0..=30.0));
                        ui.label("Edge Thickness:");
                        ui.add(egui::Slider::new(&mut self.edge_thickness, 1.0..=10.0));
                    });
                },
                VisualizationMode::Grid => {
                    ui.horizontal(|ui| {
                        ui.label("Cell Size:");
                        ui.add(egui::Slider::new(&mut self.grid_cell_size, 20.0..=100.0));
                    });
                },
                VisualizationMode::BarChart => { /* No specific config needed here */ }
            }
            
            ui.separator();
            ui.heading("Actions");
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Step").clicked() {
                    self.step_simulation();
                }
                if ui.checkbox(&mut self.auto_step, "Auto-Step").changed() {
                    // Reset timer when toggling auto-step
                    self.last_step_time = ui.input(|i| i.time);
                }
            });

            if ui.button("Run Until Stable").clicked() {
                if let Some(graph) = &mut self.graph {
                    match graph.run(self.max_steps, &mut self.rng) {
                        Ok(steps) => println!("Simulation finished in {} steps", steps),
                        Err(e) => self.error_message = Some(format!("Run error: {}", e)),
                    }
                    self.display_step = graph.history.len() - 1;
                }
            }
            
            if ui.button("Reset Configuration").clicked() {
                self.reset_graph();
            }
            
            if ui.button("Randomize Configuration").clicked() {
                self.randomize_configuration();
            }
            
            ui.separator();
            
            // Actions on selected vertex
            if let Some(vertex_idx) = self.selected_vertex {
                 ui.label(format!("Selected Vertex: {}", vertex_idx));
                 ui.horizontal(|ui| {
                    if ui.button("Add Chip").clicked() {
                        self.add_chip();
                    }
                    if ui.button("Remove Chip").clicked() {
                        self.remove_chip();
                    }
                 });
                 if ui.button("Trigger Avalanche").clicked() {
                    self.trigger_avalanche();
                 }
            } else {
                ui.label("Select a vertex in the visualization to interact.");
            }

        } else {
            ui.label("Create a graph first.");
        }
        
        // Display Error Messages
        if let Some(err) = &self.error_message {
            ui.separator();
            ui.colored_label(egui::Color32::RED, err);
        }
    }

    fn show_content(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        // Handle auto-stepping (Does this need &mut self? Yes, for step_simulation)
        if self.auto_step {
            let current_time = ui.input(|i| i.time);
            if current_time - self.last_step_time >= self.step_interval {
                self.step_simulation(); // Needs &mut self
                self.last_step_time = current_time;
            }
            ui.ctx().request_repaint(); 
        }

        // --- Visualization Area Setup ---
        let desired_size = match self.visualization_mode {
             VisualizationMode::Network => egui::vec2(500.0, 500.0),
             VisualizationMode::Grid => egui::vec2(
                 self.grid_width as f32 * self.grid_cell_size,
                 self.grid_height as f32 * self.grid_cell_size,
             ),
             VisualizationMode::BarChart => egui::vec2(ui.available_width(), 300.0),
        };
        // Allocate painter space. Bar chart doesn't strictly need this, but we need response for others.
        let (response, painter) = ui.allocate_painter(
            desired_size, 
            // Sense click only if not bar chart
            if self.visualization_mode != VisualizationMode::BarChart { egui::Sense::click_and_drag() } else { egui::Sense::hover() } 
        );

        // --- Interaction Handling (Needs &self, BEFORE borrowing graph) ---
        let mut clicked_idx = None;
        if response.clicked() && (self.visualization_mode == VisualizationMode::Network || self.visualization_mode == VisualizationMode::Grid) {
            if let Some(pos) = response.interact_pointer_pos() {
                 clicked_idx = match self.visualization_mode {
                     VisualizationMode::Network => {
                         // Check if graph exists before accessing node_positions
                         if self.graph.is_some() && !self.node_positions.is_empty() {
                             self.node_positions.iter().position(|&node_pos| {
                                 ( (response.rect.min + node_pos) - pos).length() <= self.vertex_radius
                             })
                         } else { None }
                     }
                     VisualizationMode::Grid => {
                         let relative_pos = pos - response.rect.min;
                         let grid_x = (relative_pos.x / self.grid_cell_size).floor() as usize;
                         let grid_y = (relative_pos.y / self.grid_cell_size).floor() as usize;
                         if grid_x < self.grid_width && grid_y < self.grid_height {
                              Some(grid_y * self.grid_width + grid_x)
                         } else { None }
                     }
                     _ => None, // No interaction for BarChart
                 };
            }
        }

        // --- Apply Interaction Results (Needs &mut self) ---
        if let Some(idx) = clicked_idx {
            self.selected_vertex = Some(idx);
            if self.add_chip_to_selected {
                self.add_chip(); // Mutable call OK here
            }
        }

        // --- Drawing and Status (Needs &self.graph immutable borrow) ---
        if let Some(graph) = &self.graph {
            // History slider
            if graph.history.len() > 1 {
                ui.horizontal(|ui| {
                    ui.label(format!("Step: {} / {}", self.display_step, graph.history.len() - 1));
                    // Check if display_step needs update after interaction/step
                    self.display_step = self.display_step.min(graph.history.len() - 1);
                    ui.add(egui::Slider::new(&mut self.display_step, 0..=(graph.history.len() - 1)).text("View Step"));
                });
                ui.separator();
            } else {
                self.display_step = 0; // Reset display step if history is cleared
                ui.label("Step: 0 / 0");
                ui.separator();
            }
            
            // Draw visualization (using immutable self)
            match self.visualization_mode {
                VisualizationMode::Network => self.draw_network(&painter, &response),
                VisualizationMode::Grid => self.draw_grid(&painter, &response),
                VisualizationMode::BarChart => self.draw_bar_chart(ui),
            }

            // Display status information (using immutable graph)
            ui.separator();
            ui.label(format!("Total Chips: {}", graph.total_chips()));
            if graph.is_stable() {
                 ui.colored_label(egui::Color32::GREEN, "Stable");
            } else {
                 ui.colored_label(egui::Color32::YELLOW, "Unstable");
            }
            
        } else {
            // Reset state if graph is removed
            self.selected_vertex = None;
            self.display_step = 0;
            ui.vertical_centered(|ui| {
                ui.label("No graph created yet. Use the configuration panel to create one.");
            });
        }
    }
}