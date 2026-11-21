use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let mut file = std::fs::File::open("models/Qwen3.5-9B_Q4_K_M.gguf")?;
    let content = gguf_file::Content::read(&mut file)?;

    let t = content.tensor(&mut file, "blk.0.attn_qkv.weight", &device)?;
    let dequantized = t.dequantize(&device)?;
    
    // shape is [8192, 4096]
    let dims = dequantized.dims();
    println!("Shape: {:?}", dims);
    
    // We want to check the variance of the first dimension (8192).
    // Let's compute the mean of absolute values across the input dimension (4096)
    // for each of the 8192 output channels.
    // In candle, sum over dim 1, then divide by 4096.
    let abs_t = dequantized.abs()?;
    let means = abs_t.sum_keepdim(1)? / 4096.0;
    let means = means?.flatten_all()?;
    let means_vec = means.to_vec1::<f32>()?;
    
    // If it's block split:
    // Q is 0..2048, K is 2048..4096, V is 4096..8192
    // If it's interleaved:
    // head0: q(128), k(128), v(256)
    // head1: q(128), k(128), v(256)
    
    // Let's print the mean of means for block 0..2048
    let q_mean: f32 = means_vec[0..2048].iter().sum::<f32>() / 2048.0;
    let k_mean: f32 = means_vec[2048..4096].iter().sum::<f32>() / 2048.0;
    let v_mean: f32 = means_vec[4096..8192].iter().sum::<f32>() / 4096.0;
    println!("Block-split assumption:");
    println!("  Q (0..2048) mean abs weight: {:.5}", q_mean);
    println!("  K (2048..4096) mean abs weight: {:.5}", k_mean);
    println!("  V (4096..8192) mean abs weight: {:.5}", v_mean);
    
    // Let's compute mean abs weight for the FIRST interleaved head
    // head0: q(128), k(128), v(256)
    let head0_q: f32 = means_vec[0..128].iter().sum::<f32>() / 128.0;
    let head0_k: f32 = means_vec[128..256].iter().sum::<f32>() / 128.0;
    let head0_v: f32 = means_vec[256..512].iter().sum::<f32>() / 256.0;
    
    let head1_q: f32 = means_vec[512..640].iter().sum::<f32>() / 128.0;
    let head1_k: f32 = means_vec[640..768].iter().sum::<f32>() / 128.0;
    let head1_v: f32 = means_vec[768..1024].iter().sum::<f32>() / 256.0;
    
    println!("Interleaved assumption:");
    println!("  head0_q (0..128) mean abs: {:.5}", head0_q);
    println!("  head0_k (128..256) mean abs: {:.5}", head0_k);
    println!("  head0_v (256..512) mean abs: {:.5}", head0_v);
    println!("  head1_q (512..640) mean abs: {:.5}", head1_q);
    println!("  head1_k (640..768) mean abs: {:.5}", head1_k);
    println!("  head1_v (768..1024) mean abs: {:.5}", head1_v);
    
    Ok(())
}
