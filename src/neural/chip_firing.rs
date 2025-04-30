use std::error::Error;
use std::fmt;
use rand::Rng;

use super::NeuralNetwork;

/// Error types for Chip Firing Graphs
#[derive(Debug)]
pub enum ChipFiringError {
    DimensionMismatch(String),
    InvalidGraphStructure(String),
    NegativeChips(String),
    NoActiveVertices(String),
}

impl fmt::Display for ChipFiringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChipFiringError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            ChipFiringError::InvalidGraphStructure(msg) => write!(f, "Invalid graph structure: {}", msg),
            ChipFiringError::NegativeChips(msg) => write!(f, "Negative chips: {}", msg),
            ChipFiringError::NoActiveVertices(msg) => write!(f, "No active vertices: {}", msg),
        }
    }
}

impl Error for ChipFiringError {}

/// Update mode for Chip Firing Graph dynamics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UpdateMode {
    /// Fire one active vertex at a time
    Sequential,
    /// Fire all active vertices simultaneously
    Parallel,
}

/// Vertex selection strategy for Sequential update mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VertexSelectionStrategy {
    /// Select the first active vertex
    FirstActive,
    /// Randomly select an active vertex
    RandomActive,
}

/// A graph where vertices have chips that can be fired based on certain rules.
/// 
/// Chip Firing Graphs are a type of discrete dynamical system where:
/// - Each vertex has some number of "chips"
/// - A vertex is "active" if it has at least as many chips as its degree
/// - When a vertex fires, it sends one chip along each edge to each of its neighbors
#[derive(Debug, Clone)]
pub struct ChipFiringGraph {
    /// Number of vertices in the graph
    pub num_vertices: usize,
    
    /// Adjacency matrix: adjacency[i][j] is the number of edges from vertex i to vertex j
    adjacency_matrix: Vec<Vec<u32>>,
    
    /// Current configuration (number of chips at each vertex)
    pub configuration: Vec<i32>,
    
    /// Vertex degrees (number of edges connected to each vertex)
    pub degrees: Vec<u32>,
    
    /// History of configurations after each step
    pub history: Vec<Vec<i32>>,
    
    /// Update mode (Sequential or Parallel)
    pub update_mode: UpdateMode,
    
    /// Vertex selection strategy for Sequential update mode
    pub selection_strategy: VertexSelectionStrategy,
}

impl ChipFiringGraph {
    /// Creates a new Chip Firing Graph with the given adjacency matrix and initial configuration.
    ///
    /// # Arguments
    ///
    /// * `adjacency_matrix` - The adjacency matrix of the graph. A[i][j] is the number of edges from i to j.
    /// * `initial_configuration` - The initial number of chips at each vertex.
    ///
    /// # Returns
    ///
    /// A Result containing the new ChipFiringGraph, or an error if the input is invalid.
    pub fn new(
        adjacency_matrix: Vec<Vec<u32>>,
        initial_configuration: Vec<i32>,
    ) -> Result<Self, ChipFiringError> {
        let num_vertices = adjacency_matrix.len();
        
        // Validate the adjacency matrix
        for (i, row) in adjacency_matrix.iter().enumerate() {
            if row.len() != num_vertices {
                return Err(ChipFiringError::DimensionMismatch(format!(
                    "Row {} of adjacency matrix has length {} but expected {}",
                    i, row.len(), num_vertices
                )));
            }
        }
        
        // Validate the initial configuration
        if initial_configuration.len() != num_vertices {
            return Err(ChipFiringError::DimensionMismatch(format!(
                "Initial configuration has length {} but expected {}",
                initial_configuration.len(), num_vertices
            )));
        }
        
        for (i, &chips) in initial_configuration.iter().enumerate() {
            if chips < 0 {
                return Err(ChipFiringError::NegativeChips(format!(
                    "Vertex {} has {} chips, but negative chips are not allowed",
                    i, chips
                )));
            }
        }
        
        // Calculate vertex degrees
        let mut degrees = vec![0; num_vertices];
        for i in 0..num_vertices {
            for j in 0..num_vertices {
                degrees[i] += adjacency_matrix[i][j];
            }
        }
        
        Ok(ChipFiringGraph {
            num_vertices,
            adjacency_matrix,
            configuration: initial_configuration.clone(),
            degrees,
            history: vec![initial_configuration],
            update_mode: UpdateMode::Sequential,
            selection_strategy: VertexSelectionStrategy::FirstActive,
        })
    }
    
    /// Creates a new Chip Firing Graph with a simple undirected structure (1 = edge, 0 = no edge).
    /// 
    /// # Arguments
    /// 
    /// * `edges` - A list of (from, to) vertex index pairs representing edges.
    /// * `num_vertices` - The total number of vertices in the graph.
    /// * `initial_configuration` - The initial number of chips at each vertex.
    /// 
    /// # Returns
    /// 
    /// A Result containing the new ChipFiringGraph, or an error if the input is invalid.
    pub fn from_edge_list(
        edges: &[(usize, usize)],
        num_vertices: usize,
        initial_configuration: Vec<i32>,
    ) -> Result<Self, ChipFiringError> {
        // Validate the initial configuration
        if initial_configuration.len() != num_vertices {
            return Err(ChipFiringError::DimensionMismatch(format!(
                "Initial configuration has length {} but expected {}",
                initial_configuration.len(), num_vertices
            )));
        }
        
        // Create empty adjacency matrix
        let mut adjacency_matrix = vec![vec![0; num_vertices]; num_vertices];
        
        // Fill in the adjacency matrix based on the edge list
        for &(from, to) in edges {
            if from >= num_vertices || to >= num_vertices {
                return Err(ChipFiringError::InvalidGraphStructure(format!(
                    "Edge ({}, {}) references vertex outside range 0..{}",
                    from, to, num_vertices
                )));
            }
            
            // For undirected graphs, add edges in both directions
            adjacency_matrix[from][to] += 1;
            adjacency_matrix[to][from] += 1;
        }
        
        Self::new(adjacency_matrix, initial_configuration)
    }
    
    /// Creates a new Chip Firing Graph with a pre-defined grid structure.
    /// 
    /// # Arguments
    /// 
    /// * `width` - Width of the grid.
    /// * `height` - Height of the grid.
    /// * `initial_configuration` - The initial number of chips at each vertex.
    /// 
    /// # Returns
    /// 
    /// A Result containing the new ChipFiringGraph, or an error if the input is invalid.
    pub fn new_grid(
        width: usize,
        height: usize,
        initial_configuration: Vec<i32>,
    ) -> Result<Self, ChipFiringError> {
        let num_vertices = width * height;
        
        // Validate the initial configuration
        if initial_configuration.len() != num_vertices {
            return Err(ChipFiringError::DimensionMismatch(format!(
                "Initial configuration has length {} but expected {}",
                initial_configuration.len(), num_vertices
            )));
        }
        
        // Create empty adjacency matrix
        let mut adjacency_matrix = vec![vec![0; num_vertices]; num_vertices];
        
        // Fill in the adjacency matrix with grid connections (4-connectivity)
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                
                // Connect to neighbors (up, down, left, right)
                if y > 0 { // Up
                    let up_idx = (y - 1) * width + x;
                    adjacency_matrix[idx][up_idx] = 1;
                    adjacency_matrix[up_idx][idx] = 1;
                }
                if x > 0 { // Left
                    let left_idx = y * width + (x - 1);
                    adjacency_matrix[idx][left_idx] = 1;
                    adjacency_matrix[left_idx][idx] = 1;
                }
                // No need to check down and right as they will be covered by other vertices
            }
        }
        
        Self::new(adjacency_matrix, initial_configuration)
    }
    
    /// Returns a vector of indices of currently active vertices
    /// A vertex is active if it has at least as many chips as its degree
    pub fn active_vertices(&self) -> Vec<usize> {
        let mut active = Vec::new();
        for i in 0..self.num_vertices {
            if self.configuration[i] >= self.degrees[i] as i32 {
                active.push(i);
            }
        }
        active
    }
    
    /// Check if the current configuration is stable (no active vertices)
    pub fn is_stable(&self) -> bool {
        self.active_vertices().is_empty()
    }
    
    /// Fire a specific vertex
    /// 
    /// # Arguments
    /// 
    /// * `vertex` - The index of the vertex to fire
    /// 
    /// # Returns
    /// 
    /// Result with the updated configuration or an error
    pub fn fire_vertex(&mut self, vertex: usize) -> Result<(), ChipFiringError> {
        if vertex >= self.num_vertices {
            return Err(ChipFiringError::InvalidGraphStructure(format!(
                "Vertex {} is outside valid range 0..{}", vertex, self.num_vertices
            )));
        }
        
        // Check if the vertex is active
        if self.configuration[vertex] < self.degrees[vertex] as i32 {
            return Err(ChipFiringError::NoActiveVertices(format!(
                "Vertex {} is not active: has {} chips but needs at least {} to fire",
                vertex, self.configuration[vertex], self.degrees[vertex]
            )));
        }
        
        // Update the configuration
        // The firing vertex loses one chip per outgoing edge
        self.configuration[vertex] -= self.degrees[vertex] as i32;
        
        // Each neighbor gains one chip per connecting edge
        for j in 0..self.num_vertices {
            self.configuration[j] += self.adjacency_matrix[vertex][j] as i32;
        }
        
        Ok(())
    }
    
    /// Perform one step of the dynamics based on the current update mode
    /// 
    /// # Arguments
    /// 
    /// * `rng` - Random number generator (used only for RandomActive strategy)
    /// 
    /// # Returns
    /// 
    /// Result indicating success or an error (e.g., if no vertices are active)
    pub fn step(&mut self, rng: &mut impl Rng) -> Result<(), ChipFiringError> {
        let active = self.active_vertices();
        
        if active.is_empty() {
            return Err(ChipFiringError::NoActiveVertices(
                "No active vertices to fire".to_string()
            ));
        }
        
        match self.update_mode {
            UpdateMode::Sequential => {
                // Choose which active vertex to fire based on strategy
                let vertex = match self.selection_strategy {
                    VertexSelectionStrategy::FirstActive => active[0],
                    VertexSelectionStrategy::RandomActive => {
                        active[rng.gen_range(0..active.len())]
                    }
                };
                
                self.fire_vertex(vertex)?;
            },
            UpdateMode::Parallel => {
                // Fire all active vertices simultaneously
                // We need to calculate all changes before applying them
                let mut delta = vec![0; self.num_vertices];
                
                for &vertex in &active {
                    // Vertex loses chips
                    delta[vertex] -= self.degrees[vertex] as i32;
                    
                    // Neighbors gain chips
                    for j in 0..self.num_vertices {
                        delta[j] += self.adjacency_matrix[vertex][j] as i32;
                    }
                }
                
                // Apply all changes
                for i in 0..self.num_vertices {
                    self.configuration[i] += delta[i];
                }
            }
        }
        
        // Add the new configuration to history
        self.history.push(self.configuration.clone());
        
        Ok(())
    }
    
    /// Run the dynamics for a specified number of steps or until stable
    /// 
    /// # Arguments
    /// 
    /// * `max_steps` - Maximum number of steps to run
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// 
    /// Result with the number of steps actually executed or an error
    pub fn run(
        &mut self,
        max_steps: usize,
        rng: &mut impl Rng,
    ) -> Result<usize, ChipFiringError> {
        for i in 0..max_steps {
            if self.is_stable() {
                return Ok(i); // Return if stable
            }
            
            match self.step(rng) {
                Ok(_) => {}, // Continue to next step
                Err(ChipFiringError::NoActiveVertices(_)) => {
                    return Ok(i); // Stable configuration
                },
                Err(e) => return Err(e), // Other errors
            }
        }
        
        Ok(max_steps) // Reached max steps
    }
    
    /// Add a chip to a specific vertex and run until stable
    /// This is useful for studying avalanches
    /// 
    /// # Arguments
    /// 
    /// * `vertex` - Vertex to add a chip to
    /// * `max_steps` - Maximum number of steps to run
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// 
    /// Result with the avalanche size (number of firings) or an error
    pub fn trigger_avalanche(
        &mut self,
        vertex: usize,
        max_steps: usize,
        rng: &mut impl Rng,
    ) -> Result<usize, ChipFiringError> {
        if vertex >= self.num_vertices {
            return Err(ChipFiringError::InvalidGraphStructure(format!(
                "Vertex {} is outside valid range 0..{}", vertex, self.num_vertices
            )));
        }
        
        // Add a chip to the specified vertex
        self.configuration[vertex] += 1;
        self.history.push(self.configuration.clone());
        
        // Run the dynamics
        self.run(max_steps, rng)
    }
    
    /// Calculate the total number of chips in the system
    pub fn total_chips(&self) -> i32 {
        self.configuration.iter().sum()
    }
    
    /// Clear history to save memory
    pub fn clear_history(&mut self) {
        let current = self.configuration.clone();
        self.history = vec![current];
    }
    
    /// Reset to initial configuration
    pub fn reset(&mut self) {
        if !self.history.is_empty() {
            self.configuration = self.history[0].clone();
            self.history = vec![self.configuration.clone()];
        }
    }
    
    /// Set configuration directly
    pub fn set_configuration(&mut self, configuration: Vec<i32>) -> Result<(), ChipFiringError> {
        if configuration.len() != self.num_vertices {
            return Err(ChipFiringError::DimensionMismatch(format!(
                "Configuration has length {} but expected {}",
                configuration.len(), self.num_vertices
            )));
        }
        
        for (i, &chips) in configuration.iter().enumerate() {
            if chips < 0 {
                return Err(ChipFiringError::NegativeChips(format!(
                    "Vertex {} has {} chips, but negative chips are not allowed",
                    i, chips
                )));
            }
        }
        
        self.configuration = configuration.clone();
        self.history = vec![configuration];
        
        Ok(())
    }
    
    /// Returns a vector of neighbors for a given vertex
    pub fn neighbors(&self, vertex: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        if vertex < self.num_vertices {
            for j in 0..self.num_vertices {
                if self.adjacency_matrix[vertex][j] > 0 {
                    // Consider edge count? For now, just list unique neighbors
                    if !neighbors.contains(&j) { 
                        neighbors.push(j);
                    }
                }
            }
        }
        neighbors
    }
}

// Implement the NeuralNetwork trait for ChipFiringGraph
impl NeuralNetwork for ChipFiringGraph {
    type Input = Vec<i32>;
    type Output = Vec<Vec<i32>>;
    type Error = ChipFiringError;
    
    fn forward(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        // Create a copy of self to run without modifying the original
        let mut copy = self.clone();
        copy.set_configuration(input.clone())?;
        
        let mut rng = rand::thread_rng();
        copy.run(100, &mut rng)?;
        
        Ok(copy.history)
    }
    
    fn train(&mut self, data: &[Self::Input]) -> Result<(), Self::Error> {
        // For ChipFiringGraph, "training" could mean:
        // 1. Setting up the graph structure (already done in constructor)
        // 2. Adjusting the initial configuration based on inputs
        
        if let Some(first) = data.first() {
            self.set_configuration(first.clone())?;
        }
        
        Ok(())
    }
    
    fn size(&self) -> usize {
        self.num_vertices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_create_chip_firing_graph() {
        // Create a simple 3-vertex graph with edges 0-1 and 1-2
        let edges = vec![(0, 1), (1, 2)];
        let config = vec![2, 0, 0]; // 2 chips at vertex 0
        
        let graph = ChipFiringGraph::from_edge_list(&edges, 3, config).unwrap();
        
        assert_eq!(graph.num_vertices, 3);
        assert_eq!(graph.configuration, vec![2, 0, 0]);
        assert_eq!(graph.degrees, vec![1, 2, 1]);
    }
    
    #[test]
    fn test_active_vertices() {
        // Create a simple 3-vertex graph with edges 0-1 and 1-2
        let edges = vec![(0, 1), (1, 2)];
        let config = vec![2, 0, 0]; // 2 chips at vertex 0, which has degree 1
        
        let graph = ChipFiringGraph::from_edge_list(&edges, 3, config).unwrap();
        
        let active = graph.active_vertices();
        assert_eq!(active, vec![0]); // Only vertex 0 is active (2 chips >= degree 1)
    }
    
    #[test]
    fn test_fire_vertex() {
        // Create a simple 3-vertex graph with edges 0-1 and 1-2
        let edges = vec![(0, 1), (1, 2)];
        let config = vec![2, 0, 0]; // 2 chips at vertex 0
        
        let mut graph = ChipFiringGraph::from_edge_list(&edges, 3, config).unwrap();
        
        // Fire vertex 0
        graph.fire_vertex(0).unwrap();
        
        // After firing, vertex 0 should have 1 less chip, and vertex 1 should have 1 more
        assert_eq!(graph.configuration, vec![1, 1, 0]);
    }
    
    #[test]
    fn test_step_sequential() {
        // Create a simple 3-vertex graph with edges 0-1 and 1-2
        let edges = vec![(0, 1), (1, 2)];
        let config = vec![2, 0, 0]; // 2 chips at vertex 0
        
        let mut graph = ChipFiringGraph::from_edge_list(&edges, 3, config).unwrap();
        graph.update_mode = UpdateMode::Sequential;
        
        let mut rng = thread_rng();
        graph.step(&mut rng).unwrap();
        
        // After one step, vertex 0 should have fired
        assert_eq!(graph.configuration, vec![1, 1, 0]);
    }
    
    #[test]
    fn test_is_stable() {
        // Create a simple 3-vertex graph with edges 0-1 and 1-2
        let edges = vec![(0, 1), (1, 2)];
        let config = vec![0, 0, 0]; // No chips, should be stable
        
        let graph = ChipFiringGraph::from_edge_list(&edges, 3, config).unwrap();
        
        assert!(graph.is_stable());
        
        // Create another graph with an active vertex
        let config2 = vec![2, 0, 0]; // Vertex 0 is active
        let graph2 = ChipFiringGraph::from_edge_list(&edges, 3, config2).unwrap();
        
        assert!(!graph2.is_stable());
    }
    
    #[test]
    fn test_grid_creation() {
        // Create a 2x2 grid
        let config = vec![0, 0, 0, 0];
        let graph = ChipFiringGraph::new_grid(2, 2, config).unwrap();
        
        // Check degrees: each corner should have degree 2
        assert_eq!(graph.degrees, vec![2, 2, 2, 2]);
    }
}