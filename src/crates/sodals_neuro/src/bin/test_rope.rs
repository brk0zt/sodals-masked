use candle_core::{Tensor, DType, Device, Result};

fn main() -> Result<()> {
    let dev = &Device::new_cuda(0)?;
    
    // Create [x0, x1, y0, y1] (Neox style) or [x0, y0, x1, y1] (Interleaved)
    let q = Tensor::ones((1, 48, 16, 64), DType::F32, dev)?;
    
    // cos=[c0, c1], sin=[s0, s1]
    let cos = Tensor::ones((16, 32), DType::F32, dev)?;
    let sin = Tensor::ones((16, 32), DType::F32, dev)?;
    
    let out = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
    
    println!("out: {:?}", out.to_vec2::<f32>()?);
    
    Ok(())
}
