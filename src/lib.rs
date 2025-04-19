// src/hopfield.rs (or directly in lib.rs/main.rs)

use std::error::Error;
use std::fmt;
use rand::Rng; // Need Rng for async update

// Define custom error types for clarity
#[derive(Debug)]
pub enum HopfieldError {
    DimensionMismatch(String),
    InvalidStateValue(String),
    NotPerfectSquare(String), // Added error for printing non-square grids
}

impl fmt::Display for HopfieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HopfieldError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            HopfieldError::InvalidStateValue(msg) => write!(f, "Invalid state value: {}", msg),
            HopfieldError::NotPerfectSquare(msg) => write!(f, "Grid dimension error: {}", msg),
        }
    }
}

impl Error for HopfieldError {}

/// Represents a discrete-time Hopfield Network.
#[derive(Debug, Clone)]
pub struct HopfieldNetwork {
    num_neurons: usize,
    /// Weight matrix (num_neurons x num_neurons)
    weights: Vec<Vec<f64>>,
}

impl HopfieldNetwork {
    /// Creates a new Hopfield network with a specified number of neurons.
    /// Initializes weights to zero.
    ///
    /// # Arguments
    ///
    /// * `num_neurons` - The number of neurons in the network. Must be greater than 0.
    pub fn new(num_neurons: usize) -> Self {
        if num_neurons == 0 {
            panic!("Number of neurons must be greater than 0.");
        }
        HopfieldNetwork {
            num_neurons,
            weights: vec![vec![0.0; num_neurons]; num_neurons],
        }
    }

    /// Returns the number of neurons in the network.
    pub fn size(&self) -> usize {
        self.num_neurons
    }

    /// Validates if a given vector represents a valid state (+1.0 or -1.0).
    fn validate_state(state: &[f64], expected_len: usize) -> Result<(), HopfieldError> {
        if state.len() != expected_len {
            return Err(HopfieldError::DimensionMismatch(format!(
                "State vector has length {} but expected {}",
                state.len(),
                expected_len
            )));
        }
        for &val in state {
            if val != 1.0 && val != -1.0 {
                 return Err(HopfieldError::InvalidStateValue(format!(
                     "State contains value {} which is not +1.0 or -1.0", val
                 )));
            }
        }
        Ok(())
    }

    /// Trains the network to store a set of patterns using the Hebbian rule.
    /// W_ij = sum(pattern_k[i] * pattern_k[j]) for all patterns k (i != j)
    /// W_ii is set to 0.
    /// Replaces existing weights.
    pub fn train(&mut self, patterns: &[Vec<f64>]) -> Result<(), HopfieldError> {
        if patterns.is_empty() {
             println!("Warning: Training with an empty set of patterns.");
             self.weights = vec![vec![0.0; self.num_neurons]; self.num_neurons];
             return Ok(());
        }

        for pattern in patterns {
             Self::validate_state(pattern, self.num_neurons)?;
        }

        self.weights = vec![vec![0.0; self.num_neurons]; self.num_neurons];

        for pattern in patterns {
            for i in 0..self.num_neurons {
                for j in 0..self.num_neurons {
                    if i != j {
                        self.weights[i][j] += pattern[i] * pattern[j];
                    }
                    // W_ii remains 0.0 from initialization
                }
            }
        }
        Ok(())
    }

    /// Performs a single synchronous update step.
    pub fn update_step(&self, current_state: &[f64]) -> Result<Vec<f64>, HopfieldError> {
        Self::validate_state(current_state, self.num_neurons)?;
        let mut next_state = vec![0.0; self.num_neurons];
        for i in 0..self.num_neurons {
            let mut activation_sum = 0.0;
            for j in 0..self.num_neurons {
                activation_sum += self.weights[i][j] * current_state[j];
            }
            if activation_sum > 0.0 {
                next_state[i] = 1.0;
            } else if activation_sum < 0.0 {
                next_state[i] = -1.0;
            } else {
                next_state[i] = current_state[i]; // Keep previous state if sum is zero
            }
        }
        Ok(next_state)
    }

    /// Performs a single asynchronous update step on a randomly chosen neuron.
    /// Modifies the input state directly.
    fn update_step_async(&self, state: &mut [f64], rng: &mut impl Rng) -> Result<(), HopfieldError> {
        Self::validate_state(state, self.num_neurons)?;

        let neuron_index = rng.gen_range(0..self.num_neurons);

        let mut activation_sum = 0.0;
        for j in 0..self.num_neurons {
            // Use state[j] as the network state is updated in place
            activation_sum += self.weights[neuron_index][j] * state[j]; 
        }

        let current_neuron_state = state[neuron_index];
        let new_neuron_state = if activation_sum > 0.0 {
            1.0
        } else if activation_sum < 0.0 {
            -1.0
        } else {
            current_neuron_state // Keep previous state if sum is zero
        };

        state[neuron_index] = new_neuron_state;
        Ok(())
    }

    /// Runs the network dynamics asynchronously until convergence or max iterations.
    /// An "iteration" consists of N single-neuron updates (N = num_neurons).
    pub fn run_async(
        &self,
        initial_state: &[f64],
        max_iterations: usize,
        rng: &mut impl Rng, // Need a mutable Rng
    ) -> Result<(Vec<Vec<f64>>, usize), HopfieldError> {
        Self::validate_state(initial_state, self.num_neurons)?;

        let mut states_history: Vec<Vec<f64>> = vec![initial_state.to_vec()];
        let mut current_state = initial_state.to_vec();

        for i in 0..max_iterations {
            let state_before_sweep = current_state.clone();

            // Perform N single-neuron updates for one full sweep/iteration
            for _ in 0..self.num_neurons {
                 self.update_step_async(&mut current_state, rng)?;
            }

            // Store state after the full sweep
            states_history.push(current_state.clone());

            // Check for convergence (state hasn't changed over a full sweep)
            if current_state == state_before_sweep {
                return Ok((states_history, i)); // Converged after i full sweeps
            }
        }

        // Reached max iterations
        Ok((states_history, max_iterations))
    }

    /// Runs the network dynamics until convergence or max iterations.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - The starting state vector.
    /// * `max_iterations` - Max update steps.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    ///   - A `Vec` containing the sequence of states, starting with the initial state,
    ///     up to the final state (either converged or at max_iterations).
    ///   - The number of iterations performed (0 if it converged immediately).
    ///
    /// # Errors
    ///
    /// Returns `HopfieldError::DimensionMismatch` or `HopfieldError::InvalidStateValue` 
    /// for an invalid initial state.
    pub fn run(
        &self,
        initial_state: &[f64],
        max_iterations: usize,
    ) -> Result<(Vec<Vec<f64>>, usize), HopfieldError> {
        // Validate the initial state first
        Self::validate_state(initial_state, self.num_neurons)?;

        let mut states_history: Vec<Vec<f64>> = vec![initial_state.to_vec()];
        let mut current_state = initial_state.to_vec();

        for i in 0..max_iterations {
            let next_state = self.update_step(&current_state)?;
            states_history.push(next_state.clone()); // Store the new state

            if current_state == next_state {
                // Converged
                return Ok((states_history, i)); // Return history and iterations count
            }

            current_state = next_state;
        }

        // Reached max iterations without converging
        Ok((states_history, max_iterations))
    }

    /// Calculates the energy of a given state.
    /// E = -0.5 * sum(i != j) w_ij * s_i * s_j
    pub fn energy(&self, state: &[f64]) -> Result<f64, HopfieldError> {
        Self::validate_state(state, self.num_neurons)?;

        let mut energy = 0.0;
        for i in 0..self.num_neurons {
            for j in 0..self.num_neurons {
                // The formula typically excludes i == j, and our w_ii is zero anyway.
                // But explicitly checking i != j is safer if w_ii could be non-zero.
                 if i != j {
                    energy += self.weights[i][j] * state[i] * state[j];
                 }
            }
        }
        Ok(-0.5 * energy)
    }

    /// Runs the network dynamics, printing states as ASCII art, until convergence or max iterations.
    /// Assumes the number of neurons is a perfect square for printing.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - The starting state vector.
    /// * `max_iterations` - Max update steps.
    ///
    /// # Returns
    ///
    /// A tuple containing the final state vector and the number of iterations performed.
    ///
    /// # Errors
    ///
    /// Returns `HopfieldError` for invalid initial state or if `num_neurons` isn't a perfect square.
    pub fn run_and_print_ascii(
        &self,
        initial_state: &[f64],
        max_iterations: usize,
    ) -> Result<(Vec<f64>, usize), HopfieldError> {
        Self::validate_state(initial_state, self.num_neurons)?;

        // Calculate grid dimensions, assuming a square grid
        let side_f = (self.num_neurons as f64).sqrt();
        if side_f.fract() != 0.0 {
            return Err(HopfieldError::NotPerfectSquare(format!(
                "Number of neurons ({}) is not a perfect square, cannot print as square grid.",
                self.num_neurons
            )));
        }
        let side = side_f as usize; // width and height

        println!("--- Running Hopfield Network ---");
        println!("Initial State:");
        print_state_ascii(initial_state, side, side)?;
        println!("------------------------------");

        let mut current_state = initial_state.to_vec();
        for i in 0..max_iterations {
            let next_state = self.update_step(&current_state)?;

            // Check for convergence *before* printing the next state as an iteration step
            if current_state == next_state {
                println!("Converged after {} iterations.", i);
                println!("Final State:");
                print_state_ascii(&current_state, side, side)?; // Print the stable state
                println!("------------------------------");
                return Ok((current_state, i)); // Iterations taken is i
            }

            // Print intermediate state
            println!("Iteration {}:", i + 1);
            print_state_ascii(&next_state, side, side)?;
            println!("------------------------------");

            current_state = next_state;
        }

        // Reached max iterations without converging
        println!("Reached max iterations ({}).", max_iterations);
        println!("Final State:");
        print_state_ascii(&current_state, side, side)?;
        println!("------------------------------");
        Ok((current_state, max_iterations))
    }
}

/// Helper function to print a state vector as ASCII art to the console.
/// Assumes the state vector represents a grid row by row.
/// Uses '#' for +1.0 and '.' for -1.0.
///
/// # Arguments
///
/// * `state` - The state vector slice (`&[f64]`).
/// * `width` - The width of the grid.
/// * `height` - The height of the grid.
///
/// # Errors
///
/// Returns `HopfieldError::DimensionMismatch` if `state.len()` != `width * height`.
pub fn print_state_ascii(state: &[f64], width: usize, height: usize) -> Result<(), HopfieldError> {
    if state.len() != width * height {
        return Err(HopfieldError::DimensionMismatch(format!(
            "State length {} does not match grid dimensions {}x{}",
            state.len(), width, height
        )));
    }

    println!("+{:-<width$}+", "-"); // Top border
    for y in 0..height {
        print!("|"); // Left border
        for x in 0..width {
            let index = y * width + x;
            let char_to_print = match state.get(index) {
                Some(1.0) => '#', // Or use '█' for block character
                Some(-1.0) => '.', // Or use ' ' (space)
                _ => '?', // Should not happen with validation
            };
            print!("{}", char_to_print);
        }
        println!("|"); // Right border
    }
     println!("+{:-<width$}+", "-"); // Bottom border
    Ok(())
}


// --- Example Usage ---

pub fn main() -> Result<(), HopfieldError> {
    println!("Hopfield Network ASCII Example");

    // Define grid dimensions (e.g., 5x5 for console visibility)
    // To use 24x24, change these constants.
    const WIDTH: usize = 5;
    const HEIGHT: usize = 5;
    const NUM_NEURONS: usize = WIDTH * HEIGHT; // 25

    // Define patterns as 1D vectors (+1.0 for black/on, -1.0 for white/off)
    // Example: A simple cross (+) pattern for 5x5
    let pattern_cross = vec![
        -1.0, -1.0,  1.0, -1.0, -1.0, // ..#..
        -1.0, -1.0,  1.0, -1.0, -1.0, // ..#..
         1.0,  1.0,  1.0,  1.0,  1.0, // #####
        -1.0, -1.0,  1.0, -1.0, -1.0, // ..#..
        -1.0, -1.0,  1.0, -1.0, -1.0, // ..#..
    ];

    // Example: A simple 'O' pattern for 5x5
    let pattern_o = vec![
        -1.0,  1.0,  1.0,  1.0, -1.0, // .###.
         1.0, -1.0, -1.0, -1.0,  1.0, // #...#
         1.0, -1.0, -1.0, -1.0,  1.0, // #...#
         1.0, -1.0, -1.0, -1.0,  1.0, // #...#
        -1.0,  1.0,  1.0,  1.0, -1.0, // .###.
    ];

    // You can add more patterns here (e.g., for a 24x24 grid)
    // let pattern_large = vec![...; 24 * 24];

    let patterns_to_store = vec![pattern_cross.clone(), pattern_o.clone()];

    // Create and train the network
    let mut network = HopfieldNetwork::new(NUM_NEURONS);
    println!("Training network with {} neurons...", network.size());
    network.train(&patterns_to_store)?;
    println!("Training complete.");

    // --- Test Recall ---
    println!("\n--- Test 1: Recall from Noisy Cross Pattern ---");

    // Create a noisy version of the cross pattern (flip some bits)
    let mut noisy_cross = pattern_cross.clone();
    noisy_cross[0] = 1.0;  // Flip top-left from . to #
    noisy_cross[6] = 1.0;  // Flip second row, second element from . to #
    noisy_cross[12] = -1.0; // Flip middle of cross from # to .

    let max_iterations = 10;

    // Run the network, printing states along the way
    let (_final_state_cross, _iterations_cross) = network.run_and_print_ascii(&noisy_cross, max_iterations)?;

    // Note: We don't explicitly compare _final_state_cross to pattern_cross here,
    // as the focus is on visual inspection of the printed output.


    println!("\n--- Test 2: Recall from Noisy O Pattern ---");
    let mut noisy_o = pattern_o.clone();
    noisy_o[7] = 1.0; // Flip inside top-left from . to #
    noisy_o[16] = 1.0; // Flip inside middle-left from . to #

     let (_final_state_o, _iterations_o) = network.run_and_print_ascii(&noisy_o, max_iterations)?;


    println!("\n--- Test 3: Recall starting from a stored pattern (should stabilize immediately) ---");
    let (_final_state_stable, _iterations_stable) = network.run_and_print_ascii(&pattern_o, max_iterations)?;


    Ok(())
}