pub mod hopfield;
pub mod chip_firing;

use std::error::Error;

/// Common trait for all neural network implementations
pub trait NeuralNetwork {
    type Input;
    type Output;
    type Error: Error + 'static;
    
    /// Runs the network forward pass
    fn forward(&self, input: &Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Trains the network on provided data
    fn train(&mut self, data: &[Self::Input]) -> Result<(), Self::Error>;
    
    /// Returns the size or dimension of the network
    fn size(&self) -> usize;
}