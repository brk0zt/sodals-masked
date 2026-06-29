use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, Module};
use candle_transformers::models::clip;

pub struct VisionEncoder {
    model: clip::vision_model::ClipVisionTransformer,
    projector: candle_nn::Linear,
    device: Device,
}

impl VisionEncoder {
    pub fn new(weights_path: &str, llm_hidden_size: usize, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
        };
        
        // CLIP-ViT-L/14 configuration
        let config = clip::vision_model::ClipVisionConfig {
            embed_dim: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            projection_dim: 768,
            activation: clip::text_model::Activation::QuickGelu,
            num_channels: 3,
            image_size: 224,
            patch_size: 14,
        };

        let model = clip::vision_model::ClipVisionTransformer::new(vb.pp("vision_model"), &config)?;
        
        // Linear Projector: CLIP 1024 -> LLM 4096 (Standard Qwen-VL / CLIP adapter)
        let projector = candle_nn::linear(1024, llm_hidden_size, vb.pp("vision_projector"))?;

        Ok(Self { model, projector, device: device.clone() })
    }

    pub fn encode(&self, image_bytes: &[u8]) -> Result<Tensor> {
        // PHASE 51.9: Real Image Decoding - Decode PNG and preprocess for CLIP
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Image decode error: {}", e)))?;
        
        // Resize to CLIP input size (224x224)
        let img = img.resize_to_fill(224, 224, image::imageops::FilterType::Lanczos3);
        
        // Convert to RGB tensor [3, 224, 224]
        let rgb = img.to_rgb8();
        let mut input_data = Vec::with_capacity(3 * 224 * 224);
        
        // CLIP normalization: pixel = (pixel / 255.0 - mean) / std
        // mean = [0.48145466, 0.4578275, 0.40821073]
        // std = [0.26862954, 0.26130258, 0.27577711]
        let means = [0.48145466f32, 0.4578275, 0.40821073];
        let stds = [0.26862954f32, 0.26130258, 0.27577711];
        
        for c in 0..3 {
            for y in 0..224 {
                for x in 0..224 {
                    let pixel = rgb.get_pixel(x, y)[c] as f32 / 255.0;
                    let normalized = (pixel - means[c]) / stds[c];
                    input_data.push(normalized);
                }
            }
        }
        
        let input = Tensor::from_vec(input_data, (1, 3, 224, 224), &self.device)?;
        
        // Extract CLIP features [1, 257, 1024]
        let features = self.model.forward(&input)?;
        
        // Map to LLM space [1, 257, hidden_size]
        self.projector.forward(&features)
    }
}
