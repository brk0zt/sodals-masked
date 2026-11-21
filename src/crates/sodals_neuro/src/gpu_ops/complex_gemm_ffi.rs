use candle_core::{Tensor, Result as CandleResult, Error as CandleError};
use crate::gpu_ops::memory_utils::SodalsCudaAccess;

#[cfg(feature = "cuda")]
extern "C" {
    pub fn apply_cublas_cgemm_raw(
        a_r_ptr: u64, a_i_ptr: u64,
        b_r_ptr: u64, b_i_ptr: u64,
        c_r_ptr: u64, c_i_ptr: u64,
        m: i32, n: i32, k: i32,
        trans_a: bool, trans_b: bool
    );

    pub fn bind_phasors_raw(
        a_r_ptr: u64, a_i_ptr: u64,
        b_r_ptr: u64, b_i_ptr: u64,
        out_r_ptr: u64, out_i_ptr: u64,
        size: usize
    );

    pub fn unbind_phasors_raw(
        c_r_ptr: u64, c_i_ptr: u64,
        k_r_ptr: u64, k_i_ptr: u64,
        out_r_ptr: u64, out_i_ptr: u64,
        size: usize
    );

    pub fn apply_viscosity_raw(
        p_r_ptr: u64, p_i_ptr: u64,
        impedance: f32,
        size: usize
    );
}

#[cfg(not(feature = "cuda"))]
pub fn apply_cublas_cgemm_raw(
    _a_r_ptr: u64, _a_i_ptr: u64,
    _b_r_ptr: u64, _b_i_ptr: u64,
    _c_r_ptr: u64, _c_i_ptr: u64,
    _m: i32, _n: i32, _k: i32,
    _trans_a: bool, _trans_b: bool
) {}

#[cfg(not(feature = "cuda"))]
pub fn bind_phasors_raw(
    _a_r_ptr: u64, _a_i_ptr: u64,
    _b_r_ptr: u64, _b_i_ptr: u64,
    _out_r_ptr: u64, _out_i_ptr: u64,
    _size: usize
) {}

#[cfg(not(feature = "cuda"))]
pub fn unbind_phasors_raw(
    _c_r_ptr: u64, _c_i_ptr: u64,
    _k_r_ptr: u64, _k_i_ptr: u64,
    _out_r_ptr: u64, _out_i_ptr: u64,
    _size: usize
) {}

#[cfg(not(feature = "cuda"))]
pub fn apply_viscosity_raw(
    _p_r_ptr: u64, _p_i_ptr: u64,
    _impedance: f32,
    _size: usize
) {}
