#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::{self, egui};
use raum::ui::RaumApp;

fn load_icon(path: &str) -> Result<egui::IconData, Box<dyn std::error::Error>> {
    let image = image::open(path)?.to_rgba8();
    let (width, height) = image.dimensions();
    Ok(egui::IconData {
        rgba: image.into_raw(),
        width,
        height,
    })
}

fn main() -> Result<(), eframe::Error> {
    // Initialize logger
    env_logger::init();

    let icon = load_icon("assets/icon.png")
        .expect("Failed to load application icon from assets/icon.png");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_icon(icon), // Set the window icon
        ..Default::default()
    };

    eframe::run_native(
        "Raum",
        options,
        Box::new(|cc| Ok(Box::new(RaumApp::new(cc)))),
    )
}