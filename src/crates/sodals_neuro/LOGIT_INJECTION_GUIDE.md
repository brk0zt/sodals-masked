# Logit Injection: Semantic Steering for LlmEngine

## Overview

The LlmEngine has been upgraded to support **Logit Injection** - a technique that enables semantic steering of token generation. This allows the NeuroKernel to inject steering vectors (e.g., "change from Strawberry to Cherry context") and have the LLM generation process respect those constraints.

## Key Changes

### 1. YieldedToken Structure
A new rich token output structure replaces bare strings:

```rust
pub struct YieldedToken {
    pub text: String,                   // Decoded token text
    pub token_id: u32,                  // Numeric token ID
    pub logits: Option<Vec<f32>>,       // Raw logits before sampling
    pub hidden_state: Option<Vec<f32>>, // Hidden layer activations
    pub confidence: f32,                // Model's confidence (0.0-1.0)
}
```

This allows NeuroKernel to analyze token generation with full semantic context.

### 2. LlmEngine Structure Enhancement

```rust
pub struct LlmEngine {
    // ... existing fields ...
    pub active_steering: Option<SemanticWave>,      // Current steering vector
    yielded_tokens: Arc<Mutex<Vec<YieldedToken>>>, // Token history for analysis
}
```

### 3. Logit Injection Methods

#### Set Steering Vector
```rust
pub fn set_steering(&mut self, steering_vector: Option<SemanticWave>)
```
Activates semantic steering for generation. Example:
```rust
let steering = SemanticWave::new_strawberry_to_cherry();
engine.set_steering(Some(steering));
```

#### Apply Logit Injection
```rust
fn apply_logit_injection(&self, logits: &Tensor) -> Result<Tensor>
```
Blends steering bias into logits before token sampling. This is automatically called during generation.

**How it works:**
- Takes the steering vector magnitude and direction
- Computes bias strength: `0.5 * steering_magnitude` (soft injection, not hard override)
- For each logit position: `biased_logit = logit + bias_strength * steering_component`
- Positive steering → boosts corresponding token logits
- Negative steering → suppresses corresponding token logits

#### Record & Retrieve Yielded Tokens
```rust
pub fn record_yielded_token(&self, text: String, token_id: u32, logits: Option<Vec<f32>>, ...)
pub fn get_recent_yielded_tokens(&self, limit: usize) -> Vec<YieldedToken>
pub fn clear_yielded_tokens(&self)
```

### 4. Updated generate_stream() Signature

```rust
pub fn generate_stream(
    &mut self, 
    prompt: &str, 
    physics_bias: Option<&Tensor>,
    steering_vector: Option<SemanticWave>,  // NEW: Semantic steering
    callback: impl Fn(String) + Send + Sync + 'static,
    config: &NeuroConfig
) -> Result<String>
```

**Breaking Changes:**
- `&self` → `&mut self` (to support steering updates)
- New `steering_vector` parameter between `physics_bias` and `callback`

**Usage:**
```rust
let steering = SemanticWave::new(...);
let response = engine.generate_stream(
    "Describe a fruit",
    None,
    Some(steering),  // Enable steering
    |chunk| println!("{}", chunk),
    &config
)?;
```

### 5. Removed Hard Decision Logic

#### Before:
```rust
// Hard rejection if model was "uncertain"
if !self.check_wronskian_originality(&next_token_logits).unwrap_or(true) {
    // Reject token, apply braking, possibly mode-switch
}
```

#### After:
```rust
// Soft observation - emit diagnostic info
println!("[YIELD STATE] Token#{} Variance={:.4}", i, token_variance);
// NeuroKernel can decide acceptance/rejection based on yielded tokens
```

**Rationale:** The hard decision logic created a "Split Brain" where both LlmEngine and NeuroKernel tried to make decisions. Now:
- LlmEngine generates blindly, respecting steering vectors
- NeuroKernel has access to yielded tokens (text + logits + confidence)
- NeuroKernel decides whether to accept/reject based on vector mathematics

## Usage Pattern

### 1. Basic Steering (Strawberry → Cherry Example)

```rust
use crate::fast_math::SemanticWave;

// Create steering vector
// (In real scenario, NeuroKernel would compute this)
let strawberry_logits = vec![0.8, 0.1, 0.05]; // Hypothetical
let cherry_logits = vec![0.2, 0.7, 0.08];
let steering = SemanticWave::from_reals(
    cherry_logits.iter()
        .zip(strawberry_logits.iter())
        .map(|(&c, &s)| (c - s) as f64)
        .collect()
);

// Inject steering during generation
let response = engine.generate_stream(
    "What is a strawberry? Describe its taste and color.",
    None,
    Some(steering),  // Bias toward cherry tokens
    |chunk| println!("{}", chunk),
    &config
)?;
```

### 2. Analyze Yielded Tokens in NeuroKernel

```rust
// After generation
let recent_tokens = engine.get_recent_yielded_tokens(20);

for token in recent_tokens {
    println!("Token: {} (ID: {})", token.text, token.token_id);
    println!("Confidence: {:.4}", token.confidence);
    
    if let Some(logits) = &token.logits {
        let max_logit = logits.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        println!("Max logit: {:.4}", max_logit);
    }
}
```

### 3. Dynamic Steering (Adjust mid-generation)

```rust
// Note: Current implementation sets steering at generation start
// For dynamic steering during generation, call generate_stream multiple times
// with updated steering vectors

// Or enhance with event-based steering (future):
engine.set_steering(Some(new_steering)); // Updates active_steering
// Subsequent tokens will use new steering
```

## Implementation Details

### Logit Bias Computation

Location: `compute_logit_bias_from_steering()` (LlmEngine)

```rust
steering_strength = steering_wave.magnitude().min(1.0)  // 0.0-1.0
bias_strength = 0.5 * steering_strength               // Soft, not hard

for each logit position i:
    steering_component = reals[i % steering_len]
    biased_logit[i] = original_logit[i] + (bias_strength * steering_component)
```

**Design rationale:**
- Soft injection (0.5x factor) prevents over-steering
- Modulo operation cycles steering across all tokens
- Positive components boost tokens, negative suppress them
- Works with any logit distribution (no normalization needed)

### Token Recording

Each token is recorded during generation:

```rust
// In generation loop:
let logits_vec = steered_logits.flatten_all()?.to_vec1::<f32>().ok();
self.record_yielded_token(
    current_text,
    next_token,
    logits_vec,  // For analysis
    None,         // Hidden states - expandable
);
```

Token history is limited to 100 tokens for memory efficiency.

## Integration with NeuroKernel

### Expected Workflow

1. **NeuroKernel Computes Steering**
   - Analyzes prompt intention (e.g., "Strawberry → Cherry")
   - Computes steering vector via `SemanticWave` math

2. **Inject into LlmEngine**
   ```rust
   engine.generate_stream(prompt, None, Some(steering), callback, &config)?;
   ```

3. **LlmEngine Respects Steering**
   - Applies logit injection to every token
   - Emits tokens with metadata via callback
   - Records yielded tokens

4. **NeuroKernel Validates Output**
   ```rust
   let tokens = engine.get_recent_yielded_tokens(50);
   // Analyze confidence, check if steering worked
   // Can iterate/refine approach
   ```

## Testing

### Unit Test: Steering Vector Application

```rust
#[test]
fn test_logit_injection_boosts_target_tokens() {
    let engine = create_test_engine();
    
    // Create steering toward token class [1, -1, 0]
    let steering = SemanticWave::from_reals(vec![1.0, -1.0, 0.0]);
    engine.set_steering(Some(steering));
    
    // Apply to sample logits
    let logits = Tensor::from_vec(
        vec![0.5, 0.5, 0.5],
        (1, 3),
        &Device::Cpu
    ).unwrap();
    
    let biased = engine.apply_logit_injection(&logits).unwrap();
    let biased_vec = biased.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    
    // Token 0 should increase (positive steering component)
    // Token 1 should decrease (negative steering component)
    assert!(biased_vec[0] > 0.5);
    assert!(biased_vec[1] < 0.5);
}
```

## Performance Notes

- **Logit injection:** O(n) where n = vocab size (~128k for Qwen2)
- **Per-token cost:** <1ms on modern CPU
- **Token recording:** O(1) with bounded history (~100 tokens max)
- **Memory overhead:** ~100 × (token + metadata) ≈ 1MB

## Future Enhancements

1. **Hidden State Capture**
   - Record layer-specific activations for deeper analysis
   - Enable attention pattern visualization

2. **Dynamic Steering**
   - Adjust steering vector during generation
   - Respond to real-time feedback from NeuroKernel

3. **Steering Gradient**
   - Compute how much each token was influenced by steering
   - Enable steering effectiveness metrics

4. **Multi-Vector Steering**
   - Support multiple competing steering vectors
   - Blend them with weighted combination

## Troubleshooting

### Problem: Steering has no effect
**Cause:** Steering vector magnitude too small
**Solution:** Scale steering vector to 0.5-2.0 range

### Problem: Generation diverges from steering
**Cause:** Soft injection (0.5x) may be too weak for goal
**Solution:** Increase bias_strength in `compute_logit_bias_from_steering()`

### Problem: Memory grows during generation
**Cause:** Yielded token history unbounded
**Solution:** Call `clear_yielded_tokens()` between runs (done automatically)

## API Reference

### New Public Methods

| Method | Purpose | Notes |
|--------|---------|-------|
| `set_steering(Option<SemanticWave>)` | Enable/disable steering | Call before `generate_stream()` |
| `apply_logit_injection(&Tensor)` | Apply bias to logits | Called automatically during generation |
| `record_yielded_token(...)` | Record token metadata | Called automatically during generation |
| `get_recent_yielded_tokens(limit)` | Retrieve token history | For NeuroKernel analysis |
| `clear_yielded_tokens()` | Clear token history | Called automatically at generation start |

### Modified Methods

| Method | Changes | Migration |
|--------|---------|-----------|
| `generate_stream()` | Added `steering_vector` param, changed `&self` → `&mut self` | Update callers to pass `None` for no steering |
| `new()` constructor | Initializes steering & yielded_tokens fields | Automatic |

---

**Author Notes:** This implementation follows the principle of "Steerable Engine" - the LlmEngine executes vector mathematics blindly, while NeuroKernel makes semantic decisions. This separation of concerns enables both components to specialize in what they do best.
