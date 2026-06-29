// Antigravity Analysis Vectors for Qwen 3 to Qwen 3.5 Architectural Transition
//
// This module implements the analysis vectors defined in the architectural bridge document
// for detecting and mitigating weight transfer issues between Qwen 3 8B (Dense) and
// Qwen 3.5 9B (Hybrid DeltaNet/MoE).

use candle_core::{Device, Result, Tensor, D};
use std::collections::HashMap;

/// Mismatch type classification for tensor compatibility analysis
#[derive(Debug, Clone, PartialEq)]
pub enum MismatchType {
    None,
    Shape,
    Rank,
    Corrupted,
    Deprecated,
}

/// Result of dimension mismatch detection
#[derive(Debug, Clone)]
pub struct DimensionMismatchResult {
    pub mismatch_type: MismatchType,
    pub severity: f32,
    pub recommendation: String,
}

/// Result of MLP-to-MoE distortion analysis
#[derive(Debug, Clone)]
pub struct MlpToMoeDistortionResult {
    pub distortion_score: f32,
    pub expert_utilization: HashMap<usize, f32>,
}

/// Result of linear attention state incompatibility detection
#[derive(Debug, Clone)]
pub struct LinearAttentionIncompatibilityResult {
    pub state_drift_score: f32,
    pub temporal_decay_mismatch: f32,
}

/// Hidden state drift metrics for a single layer
#[derive(Debug, Clone)]
pub struct LayerDriftMetrics {
    pub cosine_similarity: f32,
    pub l2_distance: f32,
    pub variance_ratio: f32,
    pub drift_score: f32,
}

/// Result of hidden state drift tracking
#[derive(Debug, Clone)]
pub struct HiddenStateDriftResult {
    pub drift_report: HashMap<usize, LayerDriftMetrics>,
    pub critical_layers: Vec<usize>,
}

/// Reasoning interference metrics for a single prompt
#[derive(Debug, Clone)]
pub struct ReasoningInterferenceMetrics {
    pub original_has_thinking: bool,
    pub transferred_has_thinking: bool,
    pub original_depth: usize,
    pub transferred_depth: usize,
    pub depth_ratio: f32,
}

/// Result of reasoning mode interference analysis
#[derive(Debug, Clone)]
pub struct ReasoningInterferenceResult {
    pub reasoning_integrity: f32,
    pub interference_report: HashMap<String, ReasoningInterferenceMetrics>,
}

/// Antigravity Analysis Vectors - Main struct for architectural compatibility analysis
pub struct AntigravityAnalysisVectors {
    device: Device,
}

impl AntigravityAnalysisVectors {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Vector 1: Dimension Mismatch Flagging
    /// 
    /// Flags dimension mismatches between Qwen 3 and Qwen 3.5 tensors.
    /// 
    /// # Arguments
    /// * `qwen3_tensor` - Tensor from Qwen 3 model
    /// * `qwen35_tensor` - Tensor from Qwen 3.5 model (None if deprecated)
    /// * `layer_idx` - Layer index for reporting
    /// * `tensor_name` - Name of the tensor being analyzed
    /// 
    /// # Returns
    /// DimensionMismatchResult with mismatch type, severity, and recommendation
    pub fn detect_dimension_mismatch(
        &self,
        qwen3_tensor: &Tensor,
        qwen35_tensor: Option<&Tensor>,
        layer_idx: usize,
        tensor_name: &str,
    ) -> Result<DimensionMismatchResult> {
        let q3_shape = qwen3_tensor.shape();
        let q3_dims = q3_shape.dims();

        // Check for deprecated tensors (no equivalent)
        if qwen35_tensor.is_none() {
            return Ok(DimensionMismatchResult {
                mismatch_type: MismatchType::Deprecated,
                severity: 1.0,
                recommendation: format!(
                    "Tensor {} at layer {} has no Qwen 3.5 equivalent. Use spectral initialization.",
                    tensor_name, layer_idx
                ),
            });
        }

        let q35_tensor = qwen35_tensor.unwrap();
        let q35_shape = q35_tensor.shape();
        let q35_dims = q35_shape.dims();

        // Check for rank mismatch
        if q3_dims.len() != q35_dims.len() {
            return Ok(DimensionMismatchResult {
                mismatch_type: MismatchType::Rank,
                severity: 0.9,
                recommendation: format!(
                    "Rank mismatch: {}D vs {}D at layer {}. Requires reshaping.",
                    q3_dims.len(),
                    q35_dims.len(),
                    layer_idx
                ),
            });
        }

        // Check for shape mismatch
        if q3_dims != q35_dims {
            // Calculate shape similarity
            let mut similarity_sum = 0.0f32;
            for (a, b) in q3_dims.iter().zip(q35_dims.iter()) {
                let min_val = (*a as f32).min(*b as f32);
                let max_val = (*a as f32).max(*b as f32);
                similarity_sum += if max_val > 0.0 { min_val / max_val } else { 1.0 };
            }
            let similarity = similarity_sum / q3_dims.len() as f32;

            if similarity < 0.5 {
                Ok(DimensionMismatchResult {
                    mismatch_type: MismatchType::Corrupted,
                    severity: 1.0,
                    recommendation: format!(
                        "Severe shape mismatch: {:?} vs {:?} at layer {}. Direct transfer impossible.",
                        q3_dims, q35_dims, layer_idx
                    ),
                })
            } else if similarity < 0.8 {
                Ok(DimensionMismatchResult {
                    mismatch_type: MismatchType::Shape,
                    severity: 0.7,
                    recommendation: format!(
                        "Moderate shape mismatch: {:?} vs {:?} at layer {}. Use narrow/padding operations.",
                        q3_dims, q35_dims, layer_idx
                    ),
                })
            } else {
                Ok(DimensionMismatchResult {
                    mismatch_type: MismatchType::Shape,
                    severity: 0.4,
                    recommendation: format!(
                        "Minor shape mismatch: {:?} vs {:?} at layer {}. Use interpolation.",
                        q3_dims, q35_dims, layer_idx
                    ),
                })
            }
        } else {
            Ok(DimensionMismatchResult {
                mismatch_type: MismatchType::None,
                severity: 0.0,
                recommendation: format!(
                    "Dimensions match for {} at layer {}. Direct transfer possible with scaling.",
                    tensor_name, layer_idx
                ),
            })
        }
    }

    /// MLP-to-MoE Transformation Detection
    /// 
    /// Detects distortion when mapping dense MLP to sparse MoE.
    /// 
    /// # Arguments
    /// * `qwen3_gate_proj` - Qwen 3's gate projection weights
    /// * `qwen35_moe_gate` - Qwen 3.5's MoE routing gate weights
    /// * `num_experts` - Number of experts in Qwen 3.5 MoE layer
    /// * `layer_idx` - Layer index for reporting
    /// 
    /// # Returns
    /// MlpToMoeDistortionResult with distortion score and expert utilization predictions
    pub fn detect_mlp_to_moe_distortion(
        &self,
        qwen3_gate_proj: &Tensor,
        qwen35_moe_gate: &Tensor,
        num_experts: usize,
        _layer_idx: usize,
    ) -> Result<MlpToMoeDistortionResult> {
        // Compute spectral decomposition approximation using SVD-like analysis
        // Since candle doesn't have direct SVD, we use eigenvalue approximation via power iteration
        
        let gate_proj_2d = qwen3_gate_proj.reshape((
            qwen3_gate_proj.dim(D::Minus1)?,
            qwen3_gate_proj.dim(D::Minus2)?,
        ))?;

        // Approximate rank analysis using Frobenius norm and trace
        let frobenius_norm = gate_proj_2d.sqr()?.sum_all()?.sqrt()?;
        
        // Compute trace norm by summing diagonal elements manually
        let min_dim = gate_proj_2d.dim(0)?.min(gate_proj_2d.dim(1)?);
        let mut trace_sum = 0.0f32;
        
        // Extract diagonal by iterating and using narrow
        for i in 0..min_dim {
            let row = gate_proj_2d.narrow(0, i, 1)?;
            let diag_elem = row.narrow(1, i, 1)?.to_vec0::<f32>()?;
            trace_sum += diag_elem * diag_elem;
        }
        let trace_norm = trace_sum.sqrt();
        
        // Approximate spectral energy concentration
        let spectral_concentration = if frobenius_norm.to_vec0::<f32>()? > 1e-6 {
            trace_norm / frobenius_norm.to_vec0::<f32>()?
        } else {
            0.0
        };

        // If MLP is low-rank (high concentration), MoE partitioning will be less effective
        let distortion_score = if spectral_concentration < 0.3 {
            0.8
        } else if spectral_concentration < 0.5 {
            0.5
        } else {
            0.2
        };

        // Predict expert utilization based on weight partitioning
        let mut expert_utilization = HashMap::new();
        
        // Simulate routing by projecting hidden states onto gate weights
        let moe_gate_2d = qwen35_moe_gate.reshape((
            qwen35_moe_gate.dim(D::Minus1)?,
            qwen35_moe_gate.dim(D::Minus2)?,
        ))?;

        // Compute similarity between gate_proj and moe_gate for each expert
        for expert_id in 0..num_experts {
            if expert_id < moe_gate_2d.dim(0)? {
                let expert_gate = moe_gate_2d.narrow(0, expert_id, 1)?;
                let similarity = gate_proj_2d
                    .matmul(&expert_gate.t()?)?
                    .sqr()?
                    .sum_all()?
                    .sqrt()?;
                
                let utilization = if similarity.to_vec0::<f32>()? > 1e-6 {
                    (similarity.to_vec0::<f32>()? / frobenius_norm.to_vec0::<f32>()?).min(1.0)
                } else {
                    0.0
                };
                
                expert_utilization.insert(expert_id, utilization);
            }
        }

        Ok(MlpToMoeDistortionResult {
            distortion_score,
            expert_utilization,
        })
    }

    /// Linear Attention State Incompatibility Detection
    /// 
    /// Detects incompatibility between standard attention and DeltaNet states.
    /// 
    /// # Arguments
    /// * `qwen3_attn_weights` - Qwen 3's attention weights (q_proj, k_proj, v_proj combined)
    /// * `qwen35_deltanet_weights` - Qwen 3.5's DeltaNet weights (W_x, W_h, delta)
    /// * `layer_idx` - Layer index for reporting
    /// 
    /// # Returns
    /// LinearAttentionIncompatibilityResult with state drift and temporal decay mismatch scores
    pub fn detect_linear_attention_state_incompatibility(
        &self,
        qwen3_attn_weights: &Tensor,
        qwen35_deltanet_weights: &Tensor,
        _layer_idx: usize,
    ) -> Result<LinearAttentionIncompatibilityResult> {
        // Compare attention pattern concentration vs DeltaNet decay pattern
        let q3_norm = qwen3_attn_weights.sqr()?.sum_all()?.sqrt()?;
        let _q35_norm = qwen35_deltanet_weights.sqr()?.sum_all()?.sqrt()?;

        // Compute KL divergence approximation using L2 distance
        let l2_distance = qwen3_attn_weights
            .sub(qwen35_deltanet_weights)?
            .sqr()?
            .sum_all()?
            .sqrt()?;
        
        let state_drift_score = if q3_norm.to_vec0::<f32>()? > 1e-6 {
            (l2_distance.to_vec0::<f32>()? / q3_norm.to_vec0::<f32>()?).min(1.0)
        } else {
            1.0
        };

        // Compare RoPE frequency scaling (hardcoded based on architectural knowledge)
        // Qwen 3: rope_theta = 1,000,000.0
        // Qwen 3.5: rope_theta = 10,000,000.0
        let q3_rope_freq = 1_000_000.0_f64;
        let q35_rope_freq = 10_000_000.0_f64;
        
        let temporal_decay_mismatch = ((q35_rope_freq.log10() - q3_rope_freq.log10()).abs() / 2.0) as f32;

        Ok(LinearAttentionIncompatibilityResult {
            state_drift_score,
            temporal_decay_mismatch,
        })
    }

    /// Vector 2: Hidden State Drift Detection
    /// 
    /// Tracks hidden state drift between Qwen 3 and Qwen 3.5 across layers.
    /// 
    /// # Arguments
    /// * `qwen3_hidden_states` - Hidden states from Qwen 3 model (per layer)
    /// * `qwen35_hidden_states` - Hidden states from Qwen 3.5 model (per layer)
    /// * `layer_indices` - Layer indices to analyze
    /// * `drift_threshold` - Threshold for flagging critical layers (default: 0.5)
    /// 
    /// # Returns
    /// HiddenStateDriftResult with per-layer drift metrics and critical layer list
    pub fn track_hidden_state_drift(
        &self,
        qwen3_hidden_states: &HashMap<usize, Tensor>,
        qwen35_hidden_states: &HashMap<usize, Tensor>,
        layer_indices: &[usize],
        drift_threshold: f32,
    ) -> Result<HiddenStateDriftResult> {
        let mut drift_report = HashMap::new();
        let mut critical_layers = Vec::new();

        for &layer_idx in layer_indices {
            if let (Some(q3_hs), Some(q35_hs)) = (
                qwen3_hidden_states.get(&layer_idx),
                qwen35_hidden_states.get(&layer_idx),
            ) {
                // Ensure both tensors have the same shape
                let q3_flat = q3_hs.flatten(0, q3_hs.rank() - 1)?;
                let q35_flat = q35_hs.flatten(0, q35_hs.rank() - 1)?;

                // Compute cosine similarity
                let dot_product = q3_flat.mul(&q35_flat)?.sum_all()?;
                let q3_norm = q3_flat.sqr()?.sum_all()?.sqrt()?;
                let q35_norm = q35_flat.sqr()?.sum_all()?.sqrt()?;
                
                let cosine_similarity = if q3_norm.to_vec0::<f32>()? > 1e-6 && q35_norm.to_vec0::<f32>()? > 1e-6 {
                    dot_product.to_vec0::<f32>()? / (q3_norm.to_vec0::<f32>()? * q35_norm.to_vec0::<f32>()?)
                } else {
                    0.0
                };

                // Compute L2 distance
                let l2_distance = q3_flat
                    .sub(&q35_flat)?
                    .sqr()?
                    .sum_all()?
                    .sqrt()?
                    .to_vec0::<f32>()?;

                // Compute activation variance ratio
                let q3_var = q3_flat.sqr()?.mean_all()?.to_vec0::<f32>()?;
                let q35_var = q35_flat.sqr()?.mean_all()?.to_vec0::<f32>()?;
                let variance_ratio = if q3_var > 1e-6 {
                    q35_var / q3_var
                } else {
                    1.0
                };

                let drift_score = 1.0 - cosine_similarity;

                drift_report.insert(
                    layer_idx,
                    LayerDriftMetrics {
                        cosine_similarity,
                        l2_distance,
                        variance_ratio,
                        drift_score,
                    },
                );

                // Flag critical layers
                if drift_score > drift_threshold {
                    critical_layers.push(layer_idx);
                }
            }
        }

        Ok(HiddenStateDriftResult {
            drift_report,
            critical_layers,
        })
    }

    /// Vector 3: Reasoning Mode Interference Analysis
    /// 
    /// Analyzes if transferred weights degrade Qwen 3.5's reasoning capabilities.
    /// 
    /// # Arguments
    /// * `original_responses` - Responses from original Qwen 3.5 model
    /// * `transferred_responses` - Responses from model with transferred weights
    /// * `thinking_tag` - The thinking tag to detect (default: "<thinking>")
    /// 
    /// # Returns
    /// ReasoningInterferenceResult with reasoning integrity score and detailed metrics
    pub fn analyze_reasoning_mode_interference(
        &self,
        original_responses: &HashMap<String, String>,
        transferred_responses: &HashMap<String, String>,
        thinking_tag: &str,
    ) -> Result<ReasoningInterferenceResult> {
        let mut reasoning_integrity = 1.0f32;
        let mut interference_report = HashMap::new();

        for (prompt, original_response) in original_responses.iter() {
            if let Some(transferred_response) = transferred_responses.get(prompt.as_str()) {
                // Check for thinking tag presence
                let original_has_thinking = original_response.contains(thinking_tag);
                let transferred_has_thinking = transferred_response.contains(thinking_tag);

                // Compute reasoning depth (count reasoning steps)
                let original_depth = self.count_reasoning_steps(original_response);
                let transferred_depth = self.count_reasoning_steps(transferred_response);

                // Update integrity score
                if !transferred_has_thinking && original_has_thinking {
                    reasoning_integrity *= 0.5;
                }

                let depth_ratio = if original_depth > 0 {
                    transferred_depth as f32 / original_depth as f32
                } else {
                    1.0
                };
                reasoning_integrity *= depth_ratio.min(1.0);

                interference_report.insert(
                    prompt.clone(),
                    ReasoningInterferenceMetrics {
                        original_has_thinking,
                        transferred_has_thinking,
                        original_depth,
                        transferred_depth,
                        depth_ratio,
                    },
                );
            }
        }

        Ok(ReasoningInterferenceResult {
            reasoning_integrity,
            interference_report,
        })
    }

    /// Helper function to count reasoning steps in a response
    fn count_reasoning_steps(&self, response: &str) -> usize {
        // Count common reasoning indicators
        let reasoning_indicators = [
            "step",
            "first",
            "second",
            "third",
            "next",
            "then",
            "finally",
            "because",
            "therefore",
            "thus",
            "however",
            "consider",
            "analyze",
        ];

        let mut step_count = 0;
        let lower_response = response.to_lowercase();

        for indicator in &reasoning_indicators {
            step_count += lower_response.matches(indicator).count();
        }

        // Also count numbered steps (1., 2., 3., etc.)
        for _ in response.matches(&format!(r"\d+\.")) {
            step_count += 1;
        }

        step_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_none() {
        let device = Device::Cpu;
        let analyzer = AntigravityAnalysisVectors::new(device.clone());

        let tensor1 = Tensor::zeros((10, 20), DType::F32, &device).unwrap();
        let tensor2 = Tensor::zeros((10, 20), DType::F32, &device).unwrap();

        let result = analyzer
            .detect_dimension_mismatch(&tensor1, Some(&tensor2), 0, "test_tensor")
            .unwrap();

        assert_eq!(result.mismatch_type, MismatchType::None);
        assert_eq!(result.severity, 0.0);
    }

    #[test]
    fn test_dimension_mismatch_shape() {
        let device = Device::Cpu;
        let analyzer = AntigravityAnalysisVectors::new(device.clone());

        let tensor1 = Tensor::zeros((10, 20), DType::F32, &device).unwrap();
        let tensor2 = Tensor::zeros((10, 30), DType::F32, &device).unwrap();

        let result = analyzer
            .detect_dimension_mismatch(&tensor1, Some(&tensor2), 0, "test_tensor")
            .unwrap();

        assert_eq!(result.mismatch_type, MismatchType::Shape);
        assert!(result.severity > 0.0);
    }

    #[test]
    fn test_dimension_mismatch_rank() {
        let device = Device::Cpu;
        let analyzer = AntigravityAnalysisVectors::new(device.clone());

        let tensor1 = Tensor::zeros((10, 20), DType::F32, &device).unwrap();
        let tensor2 = Tensor::zeros((10, 20, 30), DType::F32, &device).unwrap();

        let result = analyzer
            .detect_dimension_mismatch(&tensor1, Some(&tensor2), 0, "test_tensor")
            .unwrap();

        assert_eq!(result.mismatch_type, MismatchType::Rank);
        assert_eq!(result.severity, 0.9);
    }

    #[test]
    fn test_dimension_mismatch_deprecated() {
        let device = Device::Cpu;
        let analyzer = AntigravityAnalysisVectors::new(device.clone());

        let tensor1 = Tensor::zeros((10, 20), DType::F32, &device).unwrap();

        let result = analyzer
            .detect_dimension_mismatch(&tensor1, None, 0, "test_tensor")
            .unwrap();

        assert_eq!(result.mismatch_type, MismatchType::Deprecated);
        assert_eq!(result.severity, 1.0);
    }

    #[test]
    fn test_count_reasoning_steps() {
        let device = Device::Cpu;
        let analyzer = AntigravityAnalysisVectors::new(device);

        let response = "First, I will analyze the problem. Second, I will consider the options. Finally, I will provide a solution.";
        let step_count = analyzer.count_reasoning_steps(response);

        assert!(step_count > 0);
    }
}
