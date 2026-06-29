use candle_core::{Device, Tensor, DType};

fn main() {
    let device = Device::Cpu;
    
    // Input x: shape [1, 1, 2, 4]
    // 2 tokens, head_dim 4
    let x_data = vec![
        1.0f32, 2.0f32, 3.0f32, 4.0f32, // token 0
        5.0f32, 6.0f32, 7.0f32, 8.0f32, // token 1
    ];
    let x = Tensor::from_vec(x_data, (1, 1, 2, 4), &device).unwrap();
    
    // cos and sin: shape [2, 2] (d/2 = 2)
    // For token 0: cos=1, sin=0 (no rotation)
    // For token 1: cos=0, sin=1 (90 degree rotation: [a, b] -> [-b, a])
    let cos_data = vec![1.0f32, 1.0f32, 0.0f32, 0.0f32];
    let sin_data = vec![0.0f32, 0.0f32, 1.0f32, 1.0f32];
    
    let cos = Tensor::from_vec(cos_data, (2, 2), &device).unwrap();
    let sin = Tensor::from_vec(sin_data, (2, 2), &device).unwrap();
    
    let out = candle_nn::rotary_emb::rope(&x, &cos, &sin).unwrap();
    
    println!("=== ROPE TEST ===");
    println!("Input x shape: {:?}", x.shape());
    println!("Input x (flattened): {:?}", x.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    println!("Output (flattened): {:?}", out.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    println!("=================");
}
