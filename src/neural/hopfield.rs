use std::error::Error;
use std::fmt;
use rand::Rng;
use nalgebra::{DMatrix};

use super::NeuralNetwork;

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

// Enum to select the training rule
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainingRule {
    Hebbian,
    PseudoInverse,
}

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

    /// Calculates the weight matrix W based on the provided patterns ξ using the
    /// selected rule:
    /// Hebbian: W_ij = Σ_p (ξ_i^p * ξ_j^p) for i ≠ j
    /// PseudoInverse: W_ij = Σ_{α,β} ξ_i^α (C⁻¹)_{αβ} ξ_j^β where C is overlap matrix
    /// W_ii = 0
    /// This method resets existing weights before training.
    pub fn train(&mut self, patterns: &[Vec<f64>], rule: TrainingRule) -> Result<(), HopfieldError> {
        if patterns.is_empty() {
             println!("Warning: Training with an empty set of patterns.");
             self.weights = vec![vec![0.0; self.num_neurons]; self.num_neurons];
             return Ok(());
        }

        for pattern in patterns {
             Self::validate_state(pattern, self.num_neurons)?;
        }

        self.weights = vec![vec![0.0; self.num_neurons]; self.num_neurons];

        match rule {
            TrainingRule::Hebbian => {
                println!("Training using Hebbian rule...");
                for pattern in patterns {
                    for i in 0..self.num_neurons {
                        for j in 0..self.num_neurons {
                            if i != j {
                                self.weights[i][j] += pattern[i] * pattern[j];
                            }
                        }
                    }
                }
            }
            TrainingRule::PseudoInverse => {
                println!("Training using PseudoInverse rule...");
                let num_patterns = patterns.len();

                // 1. Calculate the Covariance/Overlap Matrix C
                let mut covariance_matrix = DMatrix::<f64>::zeros(num_patterns, num_patterns);
                let n_f64 = self.num_neurons as f64;

                for alpha in 0..num_patterns {
                    for beta in alpha..num_patterns {
                        let mut dot_product = 0.0;
                        for i in 0..self.num_neurons {
                            dot_product += patterns[alpha][i] * patterns[beta][i];
                        }
                        let overlap = dot_product / n_f64;
                        covariance_matrix[(alpha, beta)] = overlap;
                        if alpha != beta {
                            covariance_matrix[(beta, alpha)] = overlap;
                        }
                    }
                }

                // 2. Calculate the inverse C⁻¹
                if let Some(inv_covariance_matrix) = covariance_matrix.try_inverse() {
                    // 3. Calculate weights W_ij = Σ_{α,β} ξ_i^α (C⁻¹)_{αβ} ξ_j^β
                    for i in 0..self.num_neurons {
                        for j in 0..self.num_neurons {
                            if i == j { continue; } // W_ii = 0
                            let mut weight_sum = 0.0;
                            for alpha in 0..num_patterns {
                                for beta in 0..num_patterns {
                                    weight_sum += patterns[alpha][i] * inv_covariance_matrix[(alpha, beta)] * patterns[beta][j];
                                }
                            }
                            self.weights[i][j] = weight_sum;
                        }
                    }
                } else {
                    // Handle non-invertible matrix
                    return Err(HopfieldError::DimensionMismatch(
                        "Covariance matrix is singular, cannot compute pseudo-inverse. Try Hebbian rule or different patterns.".to_string()
                    ));
                }
            }
        }

        Ok(())
    }

    /// Performs a single synchronous update step for all neurons.
    ///
    /// Calculates the next state S(t+1) based on the current state S(t):
    /// S_i(t+1) = +1 with probability 1 / (1 + exp(-2 * β * Σ_j W_ij * S_j(t)))
    /// S_i(t+1) = -1 otherwise.
    pub fn update_step(&self, current_state: &[f64], beta: f64, rng: &mut impl Rng) -> Result<Vec<f64>, HopfieldError> {
        Self::validate_state(current_state, self.num_neurons)?;
        let mut next_state = vec![0.0; self.num_neurons];
        for i in 0..self.num_neurons {
            let mut activation_sum = 0.0;
            for j in 0..self.num_neurons {
                activation_sum += self.weights[i][j] * current_state[j];
            }
            let scaled_activation = beta * activation_sum; // Apply beta scaling

            // Calculate probability P(S_i = +1)
            // Use sigmoid: 1 / (1 + exp(-2 * x)) where x = scaled_activation
            let prob_plus_one = 1.0 / (1.0 + (-2.0 * scaled_activation).exp());

            // Decide state based on probability
            if rng.gen::<f64>() < prob_plus_one {
                 next_state[i] = 1.0;
            } else {
                 next_state[i] = -1.0;
            }
        }
        Ok(next_state)
    }

    /// Performs a single asynchronous update step on a randomly chosen neuron `k`.
    /// Modifies the input state `state` directly.
    ///
    /// Calculates the activation for neuron `k`: h_k = Σ_j W_kj * S_j
    /// Updates the state of neuron `k` probabilistically based on β * h_k.
    pub fn update_step_async(&self, state: &mut [f64], beta: f64, rng: &mut impl Rng) -> Result<(), HopfieldError> {
        Self::validate_state(state, self.num_neurons)?;

        let neuron_index = rng.gen_range(0..self.num_neurons);

        let mut activation_sum = 0.0;
        for j in 0..self.num_neurons {
            // Use state[j] as the network state is updated in place
            activation_sum += self.weights[neuron_index][j] * state[j]; 
        }

        let scaled_activation = beta * activation_sum; // Apply beta scaling

        // Calculate probability P(S_k = +1)
        let prob_plus_one = 1.0 / (1.0 + (-2.0 * scaled_activation).exp());

        // Decide state based on probability
        if rng.gen::<f64>() < prob_plus_one {
            state[neuron_index] = 1.0;
        } else {
            state[neuron_index] = -1.0;
        }

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
        beta: f64, // Add beta parameter
        rng: &mut impl Rng,
    ) -> Result<(Vec<Vec<f64>>, usize), HopfieldError> {
        Self::validate_state(initial_state, self.num_neurons)?;

        let mut states_history: Vec<Vec<f64>> = vec![initial_state.to_vec()];
        let mut current_state = initial_state.to_vec();

        for _i in 0..max_iterations {
            let _state_before_sweep = current_state.clone();

            // Perform N single-neuron updates for one full sweep/iteration
            for _ in 0..self.num_neurons {
                 self.update_step_async(&mut current_state, beta, rng)?; // Pass beta
            }

            // Store state after the full sweep
            states_history.push(current_state.clone());

            // Removed early convergence check for stochastic simulation
        }

        // Reached max iterations
        Ok((states_history, max_iterations))
    }

    /// Runs the network dynamics synchronously until convergence or max iterations.
    /// The simulation runs for max_iterations.
    pub fn run(
        &self,
        initial_state: &[f64],
        max_iterations: usize,
        beta: f64,
        rng: &mut impl Rng, // Add Rng for stochastic updates
    ) -> Result<(Vec<Vec<f64>>, usize), HopfieldError> {
        // Validate the initial state first
        Self::validate_state(initial_state, self.num_neurons)?;

        let mut states_history: Vec<Vec<f64>> = vec![initial_state.to_vec()];
        let mut current_state = initial_state.to_vec();

        for _i in 0..max_iterations {
            let next_state = self.update_step(&current_state, beta, rng)?; // Pass beta and rng
            states_history.push(next_state.clone()); // Store the new state

            // Removed early convergence check for stochastic simulation

            current_state = next_state;
        }

        // Reached max iterations without converging
        Ok((states_history, max_iterations))
    }

    /// Applies an Erdős-Rényi graph topology to the weight matrix.
    /// Each potential connection (i, j) where i != j is kept with probability `p`,
    /// otherwise W_ij and W_ji are set to 0.
    ///
    /// # Arguments
    /// * `p` - The probability of keeping a connection (0.0 <= p <= 1.0).
    /// * `rng` - A random number generator.
    pub fn apply_erdos_renyi_topology(&mut self, p: f64, rng: &mut impl Rng) {
        if !(0.0..=1.0).contains(&p) {
            eprintln!("Warning: Erdős-Rényi connectivity p must be between 0.0 and 1.0. Got {}. Skipping pruning.", p);
            return;
        }
        println!("Applying Erdős-Rényi topology with p = {}", p);

        // Iterate through the upper triangle of the matrix (excluding diagonal)
        for i in 0..self.num_neurons {
            for j in (i + 1)..self.num_neurons {
                // Decide whether to keep the connection (i, j)
                if rng.gen::<f64>() > p { 
                    // Prune the connection
                    self.weights[i][j] = 0.0;
                    self.weights[j][i] = 0.0; // Ensure symmetry
                }
                // If rng.gen::<f64>() <= p, the connection remains as calculated by train()
            }
        }
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

// Implement the NeuralNetwork trait for HopfieldNetwork
impl NeuralNetwork for HopfieldNetwork {
    type Input = Vec<f64>;
    type Output = Vec<Vec<f64>>;
    type Error = HopfieldError;
    
    // Note: The trait doesn't inherently support beta or rng.
    // We'll call the run method directly in the UI where beta/rng is known.
    // This forward implementation uses a default beta (e.g., 100.0 for near deterministic) 
    // and creates its own rng instance.
    fn forward(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        let mut rng = rand::thread_rng();
        let (states, _) = self.run(input, 100, 100.0, &mut rng)?; // High beta, internal rng
        Ok(states)
    }
    
    fn train(&mut self, data: &[Self::Input]) -> Result<(), Self::Error> {
        // Default to PseudoInverse if called via trait?
        self.train(data, TrainingRule::PseudoInverse)
    }
    
    fn size(&self) -> usize {
        self.num_neurons
    }
}