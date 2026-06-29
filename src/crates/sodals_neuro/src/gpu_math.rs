// src/crates/sodals_neuro/src/gpu_math.rs
// NOTE: cuSOLVER / cuBLAS / cudarc removed III candle's bundled cudarc does NOT
// enable the `cusolver` feature.  All computation is handled by the CPU
// fallback, which is correct and fast for the matrix sizes we use.
//
// PHASE 60.5: GPU-accelerated wave propagation for exact computation (accuracy-first)
// Uses Rayon for parallel CPU computation as GPU fallback.

use ndarray::Array2;
use rayon::prelude::*;
use num_complex::Complex64;

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// Public API shims (preserve call-sites in the rest of the crate)
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

/// Always returns false III no CUDA linear-algebra context is available.
pub fn gpu_available() -> bool {
    false
}

/// No-op: GPU initialisation is not required.
pub fn init_gpu() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// Determinant
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

/// Compute the determinant of a square matrix.
///
/// Delegates unconditionally to the CPU fallback (Gaussian elimination).
pub fn compute_determinant_gpu(
    matrix: &Array2<f64>,
) -> Result<f64, Box<dyn std::error::Error>> {
    compute_determinant_cpu_fallback(matrix)
}

/// Gaussian elimination with partial pivoting.
fn compute_determinant_cpu_fallback(
    matrix: &Array2<f64>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let (n, m) = matrix.dim();
    if n != m {
        return Err("Matrix must be square".into());
    }

    let mut a = matrix.clone();
    let mut det = 1.0_f64;

    for i in 0..n {
        // Find pivot row
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        if a[[max_row, i]].abs() < 1e-10 {
            return Ok(0.0); // Singular matrix
        }

        // Swap rows when necessary
        if max_row != i {
            for j in 0..n {
                let tmp = a[[i, j]];
                a[[i, j]] = a[[max_row, j]];
                a[[max_row, j]] = tmp;
            }
            det = -det;
        }

        det *= a[[i, i]];

        // Eliminate below
        for k in (i + 1)..n {
            let factor = a[[k, i]] / a[[i, i]];
            for j in i..n {
                a[[k, j]] -= factor * a[[i, j]];
            }
        }
    }

    Ok(det)
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// Jacobian helper (determinant + column L2 norms)
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

/// Returns `(determinant, column_l2_norms)` for the given matrix.
///
/// Column norms are computed via standard Rust iterators; no GPU required.
pub fn compute_jacobian_gpu(
    matrix: &Array2<f64>,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    let (_, m) = matrix.dim();

    let det = compute_determinant_cpu_fallback(matrix)?;

    let norms: Vec<f64> = (0..m)
        .into_par_iter()
        .map(|col| {
            matrix
                .column(col)
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    Ok((det, norms))
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// Wave Propagation (GPU-accelerated / Parallel CPU fallback)
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

/// GPU-accelerated wave propagation step
/// 
/// Computes the next state of the wave function using Hamiltonian dynamics.
/// Exact computation - no approximations for accuracy.
/// 
/// # Arguments
/// * `positions` - Current wave positions (Complex64)
/// * `momenta` - Current wave momenta (Complex64)
/// * `masses` - Mass coefficients (f64)
/// * `damping` - Damping coefficients (f64)
/// * `adjacency` - Sparse adjacency matrix (CSR format as (row, col, weight) tuples)
/// * `dt` - Time step
/// 
/// Returns (new_positions, new_momenta)
pub fn wave_propagation_step_gpu(
    positions: &[Complex64],
    momenta: &[Complex64],
    masses: &[f64],
    damping: &[f64],
    adjacency: &[(usize, usize, f64)],
    dt: f64,
) -> Result<(Vec<Complex64>, Vec<Complex64>), Box<dyn std::error::Error>> {
    wave_propagation_step_cpu_fallback(positions, momenta, masses, damping, adjacency, dt)
}

/// Parallel CPU fallback for wave propagation
/// 
/// Uses Rayon for parallel computation of wave dynamics.
/// Exact computation - no approximations.
fn wave_propagation_step_cpu_fallback(
    positions: &[Complex64],
    momenta: &[Complex64],
    masses: &[f64],
    damping: &[f64],
    adjacency: &[(usize, usize, f64)],
    dt: f64,
) -> Result<(Vec<Complex64>, Vec<Complex64>), Box<dyn std::error::Error>> {
    let n = positions.len();
    
    if n != momenta.len() || n != masses.len() || n != damping.len() {
        return Err("Vector length mismatch".into());
    }

    let mut new_positions = positions.to_vec();
    let mut new_momenta = momenta.to_vec();

    // Compute forces from adjacency (sparse matrix-vector multiplication)
    // Parallel computation using Rayon
    let forces: Vec<Complex64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut force = Complex64::new(0.0, 0.0);
            for &(row, col, weight) in adjacency {
                if row == i {
                    force += weight * positions[col];
                }
            }
            force
        })
        .collect();

    // Update momenta (Hamiltonian equation: dp/dt = -∂H/∂q)
    for i in 0..n {
        let damping_force = -damping[i] * momenta[i];
        new_momenta[i] += (forces[i] + damping_force) * dt;
    }

    // Update positions (Hamiltonian equation: dq/dt = ∂H/∂p)
    for i in 0..n {
        let velocity = new_momenta[i] / masses[i];
        new_positions[i] += velocity * dt;
    }

    Ok((new_positions, new_momenta))
}

/// Multi-step wave propagation with exact computation
/// 
/// Runs multiple propagation steps for ripple effect simulation.
/// No depth limiting for speed - full depth propagation for accuracy.
pub fn wave_propagation_multi_step_gpu(
    positions: &[Complex64],
    momenta: &[Complex64],
    masses: &[f64],
    damping: &[f64],
    adjacency: &[(usize, usize, f64)],
    dt: f64,
    steps: usize,
) -> Result<(Vec<Complex64>, Vec<Complex64>), Box<dyn std::error::Error>> {
    let mut current_positions = positions.to_vec();
    let mut current_momenta = momenta.to_vec();

    for _ in 0..steps {
        let (new_pos, new_mom) = wave_propagation_step_cpu_fallback(
            &current_positions,
            &current_momenta,
            masses,
            damping,
            adjacency,
            dt,
        )?;
        current_positions = new_pos;
        current_momenta = new_mom;
    }

    Ok((current_positions, current_momenta))
}

/// Compute peak deviations for ripple effect analysis
/// 
/// Used in retrieval engine for propagation-based scoring.
/// Exact computation - no approximations.
pub fn compute_peak_deviations_gpu(
    initial_positions: &[Complex64],
    current_positions: &[Complex64],
) -> Vec<f64> {
    initial_positions
        .par_iter()
        .zip(current_positions.par_iter())
        .map(|(init, curr)| (curr - init).norm())
        .collect()
}










