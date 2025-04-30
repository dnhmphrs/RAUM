use eframe::egui;

/// Draws a grid of cells representing a state vector
pub fn draw_grid(ui: &mut egui::Ui, state: &[f64], width: usize, height: usize, cell_size: f32) {
    // Prevent drawing if state is empty or incorrect size
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
            let cell_state = state.get(index).copied().unwrap_or(0.0);

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

/// Applies noise to a state vector by flipping bits.
/// `noise_level` is the probability (0.0 to 1.0) that any given bit is flipped.
pub fn apply_noise(state: &[f64], noise_level: f32) -> Vec<f64> {
    use rand::Rng;
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