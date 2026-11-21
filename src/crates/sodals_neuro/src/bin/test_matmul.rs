
use candle_core::{Tensor, Device, DType};

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let s = Tensor::randn(0f32, 1f32, (1, 1, 128, 128), &dev)?;
    let q = Tensor::randn(0f32, 1f32, (1, 1, 128), &dev)?;
    
    let o1 = s.matmul(&q.unsqueeze(3)?)?.squeeze(3)?;
    let o2 = q.unsqueeze(2)?.matmul(&s)?.squeeze(2)?;
    
    let diff = (&o1 - &o2)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    println!("diff: {}", diff);
    
    let rms1 = (&o1 * &o1)?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    let rms2 = (&o2 * &o2)?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    println!("rms1: {}, rms2: {}", rms1, rms2);
    
    Ok(())
}

