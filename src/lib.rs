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
///
/// Stores the network weights and provides methods for training, state updates,
/// energy calculation, and running the network dynamics.
#[derive(Debug, Clone)]
pub struct HopfieldNetwork {
    num_neurons: usize,
    /// Weight matrix (W_ij) representing connection strengths.
    /// Size: num_neurons x num_neurons. W_ii is always 0.
    weights: Vec<Vec<f64>>,
}

impl HopfieldNetwork {
    /// Creates a new Hopfield network with a specified number of neurons.
    /// Initializes weights W_ij to zero.
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

    /// Returns the number of neurons (N) in the network.
    pub fn size(&self) -> usize {
        self.num_neurons
    }

    /// Validates if a given vector represents a valid bipolar state (+1.0 or -1.0).
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

    /// Trains the network using the Hebbian learning rule (outer product rule).
    ///
    /// Calculates the weight matrix W based on the provided patterns ξ:
    /// W_ij = Σ_p (ξ_i^p * ξ_j^p) for i ≠ j
    /// W_ii = 0
    /// where p iterates over all provided patterns.
    /// This method resets existing weights before training.
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

    /// Performs a single synchronous update step for all neurons.
    ///
    /// Calculates the next state S(t+1) based on the current state S(t):
    /// S_i(t+1) = sgn( Σ_j W_ij * S_j(t) )
    /// where sgn(x) is +1 if x > 0, -1 if x < 0, and S_i(t) if x = 0.
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

    /// Performs a single asynchronous update step on a randomly chosen neuron `k`.
    /// Modifies the input state `state` directly.
    ///
    /// Calculates the activation for neuron `k`: h_k = Σ_j W_kj * S_j
    /// Updates the state of neuron `k`: S_k(new) = sgn(h_k)
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
    ///
    /// An "iteration" consists of N single-neuron updates, where N = num_neurons.
    /// Convergence occurs when the state vector remains unchanged after a full
    /// sweep of N asynchronous updates.
    pub fn run_async(
        &self,
        initial_state: &[f64],
        max_iterations: usize,
        rng: &mut impl Rng,
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

    /// Runs the network dynamics synchronously until convergence or max iterations.
    ///
    /// Convergence occurs when S(t+1) = S(t).
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

    /// Calculates the Lyapunov energy function for a given state S.
    ///
    /// E = -0.5 * Σ_{i≠j} W_ij * S_i * S_j
    /// Note: The sum implicitly excludes i=j because W_ii = 0.
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
}