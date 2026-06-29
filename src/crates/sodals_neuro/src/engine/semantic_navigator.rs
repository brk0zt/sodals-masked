// src/engine/semantic_navigator.rs
// ============================================================================
// A* SEMANTIC NAVIGATOR — Minimum Topological Resistance Pathfinder
// ============================================================================
//
// ARCHITECTURE:
//   Traverses a semantic graph where each node is a 1,000,000-bit SDR concept.
//   The heuristic is derived from popcount-based Hamming similarity:
//
//     h(n) = 1.0 - similarity(sdr(n), goal_sdr)
//
//   This creates a cosine-like admissible heuristic in the hyperdimensional
//   space. The navigator finds the path of minimum semantic resistance
//   from a start concept to a goal concept vector.
//
// MEMORY MODEL:
//   - Concept vectors are stored in the SdrArena (pre-allocated).
//   - The graph adjacency is stored as a flat edge list (no HashMap).
//   - The A* open set uses std::collections::BinaryHeap (standard).
//   - The closed set uses a dense bitvec (no HashMap).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use super::sdr_core::{SdrArena, SdrVector, SlotId};

// ============================================================================
// NODE AND EDGE TYPES
// ============================================================================

/// A node in the semantic graph. Maps 1:1 to an SdrArena SlotId.
pub type NodeId = u32;

/// A directed edge in the semantic graph with a traversal cost.
#[derive(Debug, Clone, Copy)]
pub struct SemanticEdge {
    /// Source node.
    pub from: NodeId,
    /// Destination node.
    pub to: NodeId,
    /// Edge traversal cost. In the semantic space, this represents the
    /// "cognitive effort" to transition between concepts.
    /// Typically: 1.0 - similarity(from_sdr, to_sdr), but can be
    /// manually assigned for hard-wired conceptual links.
    pub cost: f32,
}

// ============================================================================
// SEMANTIC GRAPH — Dense adjacency structure (no HashMap)
// ============================================================================

/// The semantic concept graph.
///
/// Stores adjacency as a flat edge list with a CSR-style offset index
/// for O(1) neighbor lookup. No HashMap. No BTreeMap.
///
/// Memory layout:
///   - `offsets[node]` .. `offsets[node+1]` = range into `edges` for node's outgoing edges.
///   - `edges` = flat array of (target_node, cost) pairs, sorted by source node.
pub struct SemanticGraph {
    /// CSR-style offset array. offsets.len() = num_nodes + 1.
    offsets: Vec<u32>,
    /// Flat edge targets and costs, indexed by offsets.
    edge_targets: Vec<NodeId>,
    edge_costs: Vec<f32>,
    /// Total number of nodes in the graph.
    num_nodes: u32,
}

impl SemanticGraph {
    /// Build a semantic graph from a list of edges.
    ///
    /// This is the ONLY allocation for the graph structure.
    /// After construction, neighbor lookups are O(1) index arithmetic.
    pub fn from_edges(num_nodes: u32, edges: &[SemanticEdge]) -> Self {
        // Count outgoing edges per node
        let n = num_nodes as usize;
        let mut counts = vec![0u32; n];
        for e in edges {
            if (e.from as usize) < n {
                counts[e.from as usize] += 1;
            }
        }

        // Build offset array (prefix sum)
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0u32);
        for i in 0..n {
            offsets.push(offsets[i] + counts[i]);
        }

        let total_edges = *offsets.last().unwrap() as usize;
        let mut edge_targets = vec![0u32; total_edges];
        let mut edge_costs = vec![0.0f32; total_edges];

        // Fill edges using a write-head per node
        let mut write_heads = vec![0u32; n];
        for e in edges {
            let src = e.from as usize;
            if src < n {
                let pos = (offsets[src] + write_heads[src]) as usize;
                edge_targets[pos] = e.to;
                edge_costs[pos] = e.cost;
                write_heads[src] += 1;
            }
        }

        SemanticGraph {
            offsets,
            edge_targets,
            edge_costs,
            num_nodes,
        }
    }

    /// Get the outgoing neighbors of a node as (target, cost) pairs.
    #[inline]
    pub fn neighbors(&self, node: NodeId) -> NeighborIter<'_> {
        let n = node as usize;
        if n >= self.num_nodes as usize {
            return NeighborIter {
                targets: &[],
                costs: &[],
                pos: 0,
            };
        }
        let start = self.offsets[n] as usize;
        let end = self.offsets[n + 1] as usize;
        NeighborIter {
            targets: &self.edge_targets[start..end],
            costs: &self.edge_costs[start..end],
            pos: 0,
        }
    }

    /// Number of nodes in the graph.
    #[inline]
    pub fn node_count(&self) -> u32 {
        self.num_nodes
    }
}

/// Zero-allocation iterator over a node's outgoing edges.
pub struct NeighborIter<'a> {
    targets: &'a [NodeId],
    costs: &'a [f32],
    pos: usize,
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = (NodeId, f32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.targets.len() {
            let i = self.pos;
            self.pos += 1;
            Some((self.targets[i], self.costs[i]))
        } else {
            None
        }
    }
}

// ============================================================================
// A* STATE — BinaryHeap node with f-score ordering
// ============================================================================

/// A* open-set entry. Ordered by f_score (lowest first = highest priority).
#[derive(Clone)]
struct AStarNode {
    /// The semantic graph node id.
    node: NodeId,
    /// g(n): cost from start to this node along the best known path.
    g_score: f32,
    /// f(n) = g(n) + h(n): estimated total cost through this node.
    f_score: f32,
}

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score
    }
}

impl Eq for AStarNode {}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; we want min-f_score, so reverse the ordering.
        other
            .f_score
            .partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
    }
}

// ============================================================================
// A* SEARCH RESULT
// ============================================================================

/// Result of a semantic A* search.
pub struct SemanticPath {
    /// The sequence of NodeIds from start to goal (inclusive).
    pub path: Vec<NodeId>,
    /// Total path cost (g-score of the goal node).
    pub total_cost: f32,
    /// Number of nodes expanded during the search.
    pub nodes_expanded: u32,
}

// ============================================================================
// NAVIGATOR SCRATCHPAD
// ============================================================================

/// Pre-allocated dense arrays to achieve zero dynamic allocations
/// during pathfinding over the semantic graph.
pub struct NavigatorScratchpad {
    pub g_scores: Vec<f32>,
    pub came_from: Vec<u32>,
    pub closed: Vec<u64>,
    pub open_set: BinaryHeap<AStarNode>,
}

impl NavigatorScratchpad {
    /// Pre-allocate all tracking structures to the required graph capacity.
    pub fn new(capacity: u32) -> Self {
        let n = capacity as usize;
        let closed_words = (n + 63) / 64;
        NavigatorScratchpad {
            g_scores: vec![f32::INFINITY; n],
            came_from: vec![u32::MAX; n],
            closed: vec![0u64; closed_words],
            open_set: BinaryHeap::with_capacity(n.min(4096)),
        }
    }

    /// Clear internal state for a new search while preserving allocation capacity.
    pub fn clear(&mut self) {
        self.g_scores.fill(f32::INFINITY);
        self.came_from.fill(u32::MAX);
        self.closed.fill(0);
        self.open_set.clear();
    }
}

// ============================================================================
// SEMANTIC NAVIGATOR — The A* Inference Engine
// ============================================================================

/// The Semantic Navigator: an A* pathfinder over the hyperdimensional concept graph.
///
/// Uses popcount-based Hamming similarity as the admissible heuristic.
/// Finds minimum topological resistance paths between SDR concepts.
///
/// This is the semantic FPU's "instruction pointer" — it routes inference
/// through the concept space along the path of least cognitive resistance.
pub struct SemanticNavigator<'a> {
    /// The semantic concept graph (CSR adjacency).
    graph: &'a SemanticGraph,
    /// The SDR memory arena containing concept vectors.
    arena: &'a SdrArena,
}

impl<'a> SemanticNavigator<'a> {
    /// Create a navigator over the given graph and arena.
    ///
    /// The arena must contain SDR vectors for each graph node.
    /// Node i's concept vector is at arena.get(i).
    pub fn new(graph: &'a SemanticGraph, arena: &'a SdrArena) -> Self {
        SemanticNavigator { graph, arena }
    }

    /// Compute the A* heuristic: estimated cost from `node` to the goal.
    ///
    /// h(n) = 1.0 - similarity(sdr(n), goal_sdr)
    ///
    /// This is admissible because the maximum possible similarity is 1.0
    /// (identical vectors), giving h = 0.0 (underestimate).
    /// For orthogonal random vectors, h ≈ 0.5.
    /// For antipodal vectors, h ≈ 1.0.
    #[inline]
    fn heuristic(&self, node: NodeId, goal_vector: &SdrVector) -> f32 {
        let node_vector = self.arena.get(node as usize);
        1.0 - node_vector.similarity(goal_vector)
    }

    /// Find the optimal (minimum cost) path from `start` to the node
    /// whose SDR vector is closest to `goal_vector`.
    ///
    /// The search terminates when either:
    ///   1. A node with similarity >= `similarity_threshold` is reached, OR
    ///   2. The open set is exhausted (no path exists).
    ///
    /// Uses dense arrays for g_scores and came_from (no HashMap).
    /// The closed set is a dense bitvector (1 bit per node).
    pub fn find_optimal_path(
        &self,
        start: NodeId,
        goal_vector: &SdrVector,
        similarity_threshold: f32,
        scratch: &mut NavigatorScratchpad,
    ) -> Option<SemanticPath> {
        let n = self.graph.node_count() as usize;
        if start as usize >= n {
            return None;
        }

        // NO ALLOCATIONS: Clear existing scratch space.
        scratch.clear();

        let mut nodes_expanded: u32 = 0;

        // Initialize start node
        scratch.g_scores[start as usize] = 0.0;
        let h_start = self.heuristic(start, goal_vector);
        scratch.open_set.push(AStarNode {
            node: start,
            g_score: 0.0,
            f_score: h_start,
        });

        while let Some(current) = scratch.open_set.pop() {
            let current_id = current.node as usize;

            // Check if we've reached the goal (similarity above threshold)
            let current_similarity = self.arena.get(current_id).similarity(goal_vector);
            if current_similarity >= similarity_threshold {
                // Reconstruct path
                let path = self.reconstruct_path(&scratch.came_from, current.node);
                return Some(SemanticPath {
                    path,
                    total_cost: current.g_score,
                    nodes_expanded,
                });
            }

            // Check closed set (dense bitvector)
            let word_idx = current_id / 64;
            let bit_idx = current_id % 64;
            if (scratch.closed[word_idx] >> bit_idx) & 1 == 1 {
                continue; // Already expanded
            }
            scratch.closed[word_idx] |= 1u64 << bit_idx;
            nodes_expanded += 1;

            // Skip if this entry is stale (a better path was already found)
            if current.g_score > scratch.g_scores[current_id] {
                continue;
            }

            // Expand neighbors
            for (neighbor, edge_cost) in self.graph.neighbors(current.node) {
                let neighbor_id = neighbor as usize;
                if neighbor_id >= n {
                    continue;
                }

                // Check if already in closed set
                let nw = neighbor_id / 64;
                let nb = neighbor_id % 64;
                if (scratch.closed[nw] >> nb) & 1 == 1 {
                    continue;
                }

                let tentative_g = scratch.g_scores[current_id] + edge_cost;

                if tentative_g < scratch.g_scores[neighbor_id] {
                    scratch.g_scores[neighbor_id] = tentative_g;
                    scratch.came_from[neighbor_id] = current.node;

                    let h = self.heuristic(neighbor, goal_vector);
                    scratch.open_set.push(AStarNode {
                        node: neighbor,
                        g_score: tentative_g,
                        f_score: tentative_g + h,
                    });
                }
            }
        }

        // No path found — open set exhausted
        None
    }

    /// Find the node in the graph most similar to `target_vector`.
    /// Brute-force linear scan — O(N × 15625) popcount operations.
    /// For small graphs (< 10K nodes), this is faster than any index.
    pub fn find_nearest_node(&self, target_vector: &SdrVector) -> (NodeId, f32) {
        let n = self.graph.node_count();
        let mut best_node: NodeId = 0;
        let mut best_sim: f32 = f32::NEG_INFINITY;

        for i in 0..n {
            let sim = self.arena.get(i as usize).similarity(target_vector);
            if sim > best_sim {
                best_sim = sim;
                best_node = i;
            }
        }

        (best_node, best_sim)
    }

    /// Reconstruct the path from start to goal by walking the came_from chain.
    fn reconstruct_path(&self, came_from: &[u32], goal: NodeId) -> Vec<NodeId> {
        let mut path = Vec::with_capacity(64);
        let mut current = goal;
        path.push(current);

        while came_from[current as usize] != u32::MAX {
            current = came_from[current as usize];
            path.push(current);
        }

        path.reverse();
        path
    }
}

// ============================================================================
// SEMANTIC GRAPH BUILDER — Convenient construction with auto-cost
// ============================================================================

/// Builder for constructing SemanticGraph with automatic edge costs
/// derived from SDR similarity between node vectors in the arena.
pub struct SemanticGraphBuilder {
    num_nodes: u32,
    edges: Vec<SemanticEdge>,
}

impl SemanticGraphBuilder {
    pub fn new(num_nodes: u32) -> Self {
        SemanticGraphBuilder {
            num_nodes,
            edges: Vec::with_capacity(256),
        }
    }

    /// Add a directed edge with explicit cost.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, cost: f32) -> &mut Self {
        self.edges.push(SemanticEdge { from, to, cost });
        self
    }

    /// Add a directed edge with cost = 1.0 - similarity(from, to).
    /// Uses the arena to compute popcount-based similarity.
    pub fn add_edge_auto_cost(
        &mut self,
        from: NodeId,
        to: NodeId,
        arena: &SdrArena,
    ) -> &mut Self {
        let sim = arena.get(from as usize).similarity(arena.get(to as usize));
        let cost = (1.0 - sim).max(0.001); // Minimum cost to prevent zero-weight edges
        self.edges.push(SemanticEdge { from, to, cost });
        self
    }

    /// Add bidirectional edges with auto-computed costs.
    pub fn add_bidi_edge_auto_cost(
        &mut self,
        a: NodeId,
        b: NodeId,
        arena: &SdrArena,
    ) -> &mut Self {
        let sim = arena.get(a as usize).similarity(arena.get(b as usize));
        let cost = (1.0 - sim).max(0.001);
        self.edges.push(SemanticEdge { from: a, to: b, cost });
        self.edges.push(SemanticEdge { from: b, to: a, cost });
        self
    }

    /// Build the final SemanticGraph (CSR layout).
    pub fn build(self) -> SemanticGraph {
        SemanticGraph::from_edges(self.num_nodes, &self.edges)
    }
}
