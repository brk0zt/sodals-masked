#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_qkv_projection_uses_dynamic_tail_chunks() {
        let (q_chunk, k_chunk, v_chunk) = QAttention::split_qkv_projection(8192, 4096, 1024);
        assert_eq!((q_chunk, k_chunk, v_chunk), (4096, 1024, 1024));
    }

    #[test]
    fn split_qkv_projection_handles_shorter_q_chunk() {
        let (q_chunk, k_chunk, v_chunk) = QAttention::split_qkv_projection(6144, 4096, 1024);
        assert_eq!((q_chunk, k_chunk, v_chunk), (4096, 1024, 1024));
    }
}
