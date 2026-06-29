// src/crates/sodals_neuro/src/arc_agi_solver.rs
// ============================================================================
// SODALS ARC-AGI-3 A* GUIDED SOLVER
// ============================================================================

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use num_complex::Complex64;

use crate::semantic_network::ResonanceNetwork;
use crate::arkhe_pipeline::{ArkheCognitionPipeline, CliffordBridge};
use crate::semantic_parser::SemanticRegisterFile;
use crate::fast_math::SemanticWave;

#[derive(Clone, Debug)]
pub struct ArcAgiNode {
    pub grid: Vec<Vec<u32>>,
    pub path: Vec<String>,
    pub g_cost: f64, // Path cost accumulated so far via RK4 pondering
    pub h_cost: f64, // Heuristic cost based on CliffordBridge h_phase
    pub f_cost: f64, // Total cost f = g + h
}

impl PartialEq for ArcAgiNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost == other.f_cost
    }
}

impl Eq for ArcAgiNode {}

impl PartialOrd for ArcAgiNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse order for min-heap
        other.f_cost.partial_cmp(&self.f_cost)
    }
}

impl Ord for ArcAgiNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct ArcAgiSolver {
    pub max_expansions: usize,
}

impl ArcAgiSolver {
    pub fn new(max_expansions: usize) -> Self {
        Self { max_expansions }
    }

    /// RK4 integration step for cost prediction:
    /// dy/dt = f(e, ld) = a*e^2 + b*l + c*d^2
    fn rk4_ponder_cost(
        a: f64, b: f64, c: f64,
        energy: f64,
        path_len: f64,
        distance: f64,
        dt: f64,
    ) -> f64 {
        let f_val = |e: f64, l: f64, d: f64| -> f64 {
            a * e.powi(2) + b * l + c * d.powi(2)
        };
        
        let k1 = f_val(energy, path_len, distance);
        let k2 = f_val(energy + dt * 0.1, path_len + dt * 0.5, distance - dt * 0.1);
        let k3 = f_val(energy + dt * 0.1, path_len + dt * 0.5, distance - dt * 0.1);
        let k4 = f_val(energy + dt * 0.2, path_len + dt * 1.0, distance - dt * 0.2);
        
        ((k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0 * dt).max(0.1)
    }

    /// Convert a grid state to a ResonanceNetwork representation
    fn grid_to_network(grid: &[Vec<u32>]) -> ResonanceNetwork {
        let rows = grid.len();
        let cols = if rows > 0 { grid[0].len() } else { 0 };
        let n = (rows * cols).max(1);
        let mut net = ResonanceNetwork::new(n + 16);
        
        for r in 0..rows {
            for c in 0..cols {
                let color = grid[r][c];
                let label = format!("cell_{}_{}_{}", r, c, color);
                
                // Deterministic VSA signature based on coordinates and color
                let mut sig = vec![Complex64::new(0.0, 0.0); 64];
                for j in 0..64 {
                    let phase = ((r * j + c) as f64 * 0.2) + (color as f64 * std::f64::consts::FRAC_PI_3);
                    sig[j] = Complex64::from_polar(1.0, phase);
                }
                net.add_labeled_concept(&label, 0.1 * (color + 1) as f64, sig);
            }
        }
        
        for r in 0..rows {
            for c in 0..cols {
                let id = r * cols + c;
                if c + 1 < cols {
                    net.connect_semantic(id, id + 1, 1.0);
                }
                if r + 1 < rows {
                    net.connect_semantic(id, id + cols, 1.0);
                }
            }
        }
        
        net
    }

    /// A* Solver with Arkhe-Guided Heuristic and Wronskian constraints
    pub fn solve(
        &mut self,
        input_grid: &[Vec<u32>],
        target_grid: &[Vec<u32>],
        net_f: &ResonanceNetwork,
        pipeline: &mut ArkheCognitionPipeline,
        srf: &mut SemanticRegisterFile,
    ) -> Result<Vec<String>, String> {
        let mut open_set = BinaryHeap::new();
        let mut visited = HashSet::new();

        // Initial polynomial coefficients
        let mut coeff_a = 0.5;
        let mut coeff_b = 0.2;
        let mut coeff_c = 0.8;

        // Compute initial heuristic
        let mut net_init = Self::grid_to_network(input_grid);
        let impedance_theta = if net_init.n > 0 {
            net_init.damping.iter().take(net_init.n).sum::<f64>() / net_init.n as f64
        } else {
            1.0
        };
        let (h_phase, _) = pipeline.bridge.project(net_f, &net_init, impedance_theta);
        let h_cost = (std::f64::consts::PI - h_phase.abs()).max(0.0);

        open_set.push(ArcAgiNode {
            grid: input_grid.to_vec(),
            path: Vec::new(),
            g_cost: 0.0,
            h_cost,
            f_cost: h_cost,
        });

        let mut expansions = 0;

        while let Some(node) = open_set.pop() {
            if node.grid == target_grid {
                eprintln!("[ARC SOLVER] Goal reached in {} expansions!", expansions);
                return Ok(node.path);
            }

            if visited.contains(&node.grid) {
                continue;
            }
            visited.insert(node.grid.clone());

            expansions += 1;
            if expansions > self.max_expansions {
                return Err(format!("Max expansions ({}) exceeded", self.max_expansions));
            }

            // Generate neighbor states
            let neighbors = self.get_neighbors(&node.grid);
            for (action_name, neighbor_grid) in neighbors {
                if visited.contains(&neighbor_grid) {
                    continue;
                }

                // Construct state network
                let mut net_node = Self::grid_to_network(&neighbor_grid);

                // Update LSQ coefficients using VSA similarity score from Benzerlik Arkhesi
                pipeline.arkhe_benzerlik(net_f, &net_node);
                if let Some((_, _, score)) = pipeline.state.top_similarity {
                    coeff_a = (coeff_a * 0.9 + (1.0 - score) * 0.1).clamp(0.01, 2.0);
                    coeff_b = (coeff_b * 0.9 + score * 0.1).clamp(0.01, 2.0);
                    coeff_c = (coeff_c * 0.9 + (1.0 - score) * 2.0 * 0.1).clamp(0.01, 4.0);
                }

                // Run physics steps to populate dynamics
                net_node.step(&[]);

                // Compute CliffordBridge projection
                let (h_phase_node, _) = pipeline.bridge.project(net_f, &net_node, impedance_theta);

                // Check Wronskian and constructive/destructive interference (Uyum Arkhesi)
                pipeline.run(net_f, &net_node, &action_name, srf);
                let wronskian = pipeline.state.wronskian;
                let compatibility = pipeline.state.compatibility_score;

                // Budama (Pruning): Wronskian sıfıra yakınsa (redundant) veya God's Gate kapalıysa
                if wronskian < 0.001 || !pipeline.state.gate_open || compatibility < 0.1 {
                    continue;
                }

                // Geodesic trajectory length using warped Riemannian space
                let warp_map = net_node.compute_rope_warp_map(&action_name);
                let start_idx = 0;
                let target_idx = net_node.n.saturating_sub(1);
                let traj = net_node.find_hamiltonian_trajectory_warped(&[start_idx], &[target_idx], 25, &warp_map);
                let path_len = traj.len() as f64;

                let energy = net_node.compute_total_energy();
                let distance = h_phase_node.abs();

                // Compute cost via RK4 pondering integration
                let edge_cost = Self::rk4_ponder_cost(coeff_a, coeff_b, coeff_c, energy, path_len, distance, 1.0);
                let new_g_cost = node.g_cost + edge_cost;
                
                let new_h_cost = (std::f64::consts::PI - h_phase_node.abs()).max(0.0);
                let new_f_cost = new_g_cost + new_h_cost;

                let mut new_path = node.path.clone();
                new_path.push(action_name);

                open_set.push(ArcAgiNode {
                    grid: neighbor_grid,
                    path: new_path,
                    g_cost: new_g_cost,
                    h_cost: new_h_cost,
                    f_cost: new_f_cost,
                });
            }
        }

        Err("No path found to target grid".to_string())
    }

    /// Retrieve neighbor grid states by applying transformations
    fn get_neighbors(&self, grid: &[Vec<u32>]) -> Vec<(String, Vec<Vec<u32>>)> {
        let mut neighbors = Vec::new();

        // 1. Rotate 90
        neighbors.push(("rotate_90".to_string(), rotate_90(grid)));

        // 2. Flip vertical
        neighbors.push(("flip_v".to_string(), flip_v(grid)));

        // 3. Flip horizontal
        neighbors.push(("flip_h".to_string(), flip_h(grid)));

        // 4. Invert colors
        neighbors.push(("invert_colors".to_string(), invert_colors(grid)));

        // 5. Shift Up
        neighbors.push(("shift_up".to_string(), shift_up(grid)));

        // 6. Shift Down
        neighbors.push(("shift_down".to_string(), shift_down(grid)));

        // 7. Shift Left
        neighbors.push(("shift_left".to_string(), shift_left(grid)));

        // 8. Shift Right
        neighbors.push(("shift_right".to_string(), shift_right(grid)));

        // 9. Crop
        neighbors.push(("crop".to_string(), crop(grid)));

        // 10. Rotate 180
        neighbors.push(("rotate_180".to_string(), rotate_180(grid)));

        // 11. Rotate 270
        neighbors.push(("rotate_270".to_string(), rotate_270(grid)));

        // 12. Symmetry X
        neighbors.push(("symmetry_x".to_string(), symmetry_x(grid)));

        // 13. Symmetry Y
        neighbors.push(("symmetry_y".to_string(), symmetry_y(grid)));

        // 14. Symmetry Diagonal
        neighbors.push(("symmetry_diag".to_string(), symmetry_diag(grid)));

        // 15. Expand
        neighbors.push(("expand".to_string(), expand(grid)));

        // 16. Contract
        neighbors.push(("contract".to_string(), contract(grid)));

        // 17. Color Flood
        neighbors.push(("color_flood".to_string(), color_flood(grid)));

        // 18. Connect Components
        neighbors.push(("connect".to_string(), connect_components(grid)));

        // 19. Scale 2x
        neighbors.push(("scale_2x".to_string(), scale_2x(grid)));

        // 20. Recolor Dominant
        neighbors.push(("recolor_dominant".to_string(), recolor_dominant(grid)));

        neighbors
    }
}

// ============================================================================
// GRID TRANSFORMATION HELPERS
// ============================================================================

fn rotate_90(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let r = grid.len();
    let c = grid[0].len();
    let mut out = vec![vec![0; r]; c];
    for i in 0..r {
        for j in 0..c {
            out[j][r - 1 - i] = grid[i][j];
        }
    }
    out
}

fn rotate_180(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    rotate_90(&rotate_90(grid))
}

fn rotate_270(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    rotate_90(&rotate_90(&rotate_90(grid)))
}

fn flip_h(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut out = grid.to_vec();
    for row in out.iter_mut() {
        row.reverse();
    }
    out
}

fn flip_v(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut out = grid.to_vec();
    out.reverse();
    out
}

fn crop(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let r = grid.len();
    let c = grid[0].len();
    let mut r_min = r;
    let mut r_max = 0;
    let mut c_min = c;
    let mut c_max = 0;
    let mut has_non_zero = false;
    for i in 0..r {
        for j in 0..c {
            if grid[i][j] != 0 {
                has_non_zero = true;
                r_min = r_min.min(i);
                r_max = r_max.max(i);
                c_min = c_min.min(j);
                c_max = c_max.max(j);
            }
        }
    }
    if !has_non_zero {
        return vec![vec![0]];
    }
    let mut out = Vec::new();
    for i in r_min..=r_max {
        out.push(grid[i][c_min..=c_max].to_vec());
    }
    out
}

fn invert_colors(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut out = grid.to_vec();
    for row in out.iter_mut() {
        for val in row.iter_mut() {
            if *val != 0 {
                *val = (10 - *val).min(9).max(1);
            }
        }
    }
    out
}

fn shift_up(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let mut out = grid.to_vec();
    let first = out.remove(0);
    out.push(first);
    out
}

fn shift_down(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let mut out = grid.to_vec();
    if let Some(last) = out.pop() {
        out.insert(0, last);
    }
    out
}

fn shift_left(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let mut out = grid.to_vec();
    for row in out.iter_mut() {
        if !row.is_empty() {
            let first = row.remove(0);
            row.push(first);
        }
    }
    out
}

fn shift_right(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let mut out = grid.to_vec();
    for row in out.iter_mut() {
        if !row.is_empty() {
            if let Some(last) = row.pop() {
                row.insert(0, last);
            }
        }
    }
    out
}

fn symmetry_x(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let h = grid.len();
    let w = grid[0].len();
    let mut out = grid.to_vec();
    for r in 0..h / 2 {
        for c in 0..w {
            out[h - 1 - r][c] = grid[r][c];
        }
    }
    out
}

fn symmetry_y(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let h = grid.len();
    let w = grid[0].len();
    let mut out = grid.to_vec();
    for r in 0..h {
        for c in 0..w / 2 {
            out[r][w - 1 - c] = grid[r][c];
        }
    }
    out
}

fn symmetry_diag(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let h = grid.len();
    let w = grid[0].len();
    let mut out = vec![vec![0; h]; w];
    for r in 0..h {
        for c in 0..w {
            if r < w && c < h {
                out[c][r] = grid[r][c];
            }
        }
    }
    out
}

fn expand(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let h = grid.len();
    let w = grid[0].len();
    let mut out = grid.to_vec();
    for r in 0..h {
        for c in 0..w {
            if grid[r][c] != 0 {
                for &(dr, dc) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                        let nr = nr as usize;
                        let nc = nc as usize;
                        if grid[nr][nc] == 0 {
                            out[nr][nc] = grid[r][c];
                        }
                    }
                }
            }
        }
    }
    out
}

fn contract(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let h = grid.len();
    let w = grid[0].len();
    let mut out = grid.to_vec();
    for r in 0..h {
        for c in 0..w {
            if grid[r][c] != 0 {
                let mut touch_zero = false;
                for &(dr, dc) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                        if grid[nr as usize][nc as usize] == 0 {
                            touch_zero = true;
                            break;
                        }
                    } else {
                        touch_zero = true;
                        break;
                    }
                }
                if touch_zero {
                    out[r][c] = 0;
                }
            }
        }
    }
    out
}

fn color_flood(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let h = grid.len();
    let w = grid[0].len();
    let mut out = grid.to_vec();
    let mut visited = vec![vec![false; w]; h];

    for r in 0..h {
        for c in 0..w {
            if grid[r][c] == 0 && !visited[r][c] {
                let mut comp = Vec::new();
                let mut queue = vec![(r, c)];
                visited[r][c] = true;
                let mut touches_boundary = false;
                let mut border_colors = HashMap::new();

                let mut head = 0;
                while head < queue.len() {
                    let (cr, cc) = queue[head];
                    head += 1;
                    comp.push((cr, cc));

                    if cr == 0 || cr == h - 1 || cc == 0 || cc == w - 1 {
                        touches_boundary = true;
                    }

                    for &(dr, dc) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                        let nr = cr as i32 + dr;
                        let nc = cc as i32 + dc;
                        if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                            let nr = nr as usize;
                            let nc = nc as usize;
                            if grid[nr][nc] == 0 {
                                if !visited[nr][nc] {
                                    visited[nr][nc] = true;
                                    queue.push((nr, nc));
                                }
                            } else {
                                *border_colors.entry(grid[nr][nc]).or_insert(0) += 1;
                            }
                        }
                    }
                }

                if !touches_boundary && !border_colors.is_empty() {
                    let mut fill_color = 0;
                    let mut max_count = 0;
                    for (&col, &count) in &border_colors {
                        if count > max_count {
                            max_count = count;
                            fill_color = col;
                        }
                    }
                    if fill_color != 0 {
                        for &(cr, cc) in &comp {
                            out[cr][cc] = fill_color;
                        }
                    }
                }
            }
        }
    }
    out
}

fn connect_components(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let h = grid.len();
    let w = grid[0].len();
    let mut out = grid.to_vec();

    let mut color_points: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    for r in 0..h {
        for c in 0..w {
            let color = grid[r][c];
            if color > 0 {
                color_points.entry(color).or_insert(Vec::new()).push((r, c));
            }
        }
    }

    for (&color, pts) in &color_points {
        if pts.len() < 2 { continue; }
        for i in 0..pts.len() {
            for j in i+1..pts.len() {
                let (r1, c1) = pts[i];
                let (r2, c2) = pts[j];
                if r1 == r2 {
                    let min_c = c1.min(c2);
                    let max_c = c1.max(c2);
                    let mut path_clear = true;
                    for c in min_c + 1..max_c {
                        if grid[r1][c] != 0 && grid[r1][c] != color {
                            path_clear = false;
                            break;
                        }
                    }
                    if path_clear {
                        for c in min_c + 1..max_c {
                            out[r1][c] = color;
                        }
                    }
                } else if c1 == c2 {
                    let min_r = r1.min(r2);
                    let max_r = r1.max(r2);
                    let mut path_clear = true;
                    for r in min_r + 1..max_r {
                        if grid[r][c1] != 0 && grid[r][c1] != color {
                            path_clear = false;
                            break;
                        }
                    }
                    if path_clear {
                        for r in min_r + 1..max_r {
                            out[r][c1] = color;
                        }
                    }
                }
            }
        }
    }
    out
}

fn scale_2x(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let h = grid.len();
    let w = grid[0].len();
    let mut out = vec![vec![0; w * 2]; h * 2];
    for r in 0..h {
        for c in 0..w {
            let color = grid[r][c];
            out[r * 2][c * 2] = color;
            out[r * 2 + 1][c * 2] = color;
            out[r * 2][c * 2 + 1] = color;
            out[r * 2 + 1][c * 2 + 1] = color;
        }
    }
    out
}

fn recolor_dominant(grid: &[Vec<u32>]) -> Vec<Vec<u32>> {
    if grid.is_empty() { return Vec::new(); }
    let mut out = grid.to_vec();
    let mut counts = HashMap::new();
    for row in grid {
        for &val in row {
            if val != 0 {
                *counts.entry(val).or_insert(0) += 1;
            }
        }
    }
    if counts.is_empty() { return out; }
    let mut sorted_colors: Vec<_> = counts.into_iter().collect();
    sorted_colors.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    let dom_color = sorted_colors[0].0;
    
    let target_color = (dom_color % 9) + 1;
    for row in out.iter_mut() {
        for val in row.iter_mut() {
            if *val == dom_color {
                *val = target_color;
            }
        }
    }
    out
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_network(n: usize) -> ResonanceNetwork {
        let mut net = ResonanceNetwork::new(n + 4);
        for i in 0..n {
            let sig: Vec<Complex64> = (0..64)
                .map(|j| Complex64::from_polar(1.0, (i * j) as f64 * 0.1))
                .collect();
            net.add_labeled_concept(&format!("concept_{}", i), 0.05 * i as f64, sig);
        }
        net
    }

    #[test]
    fn test_grid_transforms() {
        let grid = vec![vec![1, 2], vec![3, 4]];
        let rot = rotate_90(&grid);
        assert_eq!(rot, vec![vec![3, 1], vec![4, 2]]);
        
        let fh = flip_h(&grid);
        assert_eq!(fh, vec![vec![2, 1], vec![4, 3]]);
    }

    #[test]
    fn test_new_transforms() {
        // Test rotate_180 and rotate_270
        let grid = vec![vec![1, 2], vec![3, 4]];
        assert_eq!(rotate_180(&grid), vec![vec![4, 3], vec![2, 1]]);
        assert_eq!(rotate_270(&grid), vec![vec![2, 4], vec![1, 3]]);

        // Test symmetry_x and symmetry_y
        let sym_grid = vec![vec![1, 2], vec![0, 0]];
        assert_eq!(symmetry_x(&sym_grid), vec![vec![1, 2], vec![1, 2]]);
        let sym_grid_y = vec![vec![1, 0], vec![3, 0]];
        assert_eq!(symmetry_y(&sym_grid_y), vec![vec![1, 1], vec![3, 3]]);

        // Test scale_2x
        let scale_grid = vec![vec![5]];
        assert_eq!(scale_2x(&scale_grid), vec![vec![5, 5], vec![5, 5]]);

        // Test expand and contract
        let exp_grid = vec![vec![1, 0, 0], vec![0, 0, 0], vec![0, 0, 0]];
        let expanded = expand(&exp_grid);
        assert_eq!(expanded[0][1], 1);
        assert_eq!(expanded[1][0], 1);
        assert_eq!(expanded[1][1], 0);

        // Test color_flood
        let flood_grid = vec![
            vec![1, 1, 1],
            vec![1, 0, 1],
            vec![1, 1, 1]
        ];
        assert_eq!(color_flood(&flood_grid), vec![
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![1, 1, 1]
        ]);
    }

    #[test]
    fn test_solver_integration() {
        let mut solver = ArcAgiSolver::new(50);
        let input_grid = vec![vec![1, 2], vec![3, 4]];
        let target_grid = vec![vec![3, 1], vec![4, 2]]; // rot90

        let net_f = make_test_network(4);
        let mut pipeline = ArkheCognitionPipeline::new();
        let mut srf = SemanticRegisterFile::new(0.0);

        let path = solver.solve(&input_grid, &target_grid, &net_f, &mut pipeline, &mut srf);
        assert!(path.is_ok());
        assert_eq!(path.unwrap(), vec!["rotate_90".to_string()]);
    }
}
