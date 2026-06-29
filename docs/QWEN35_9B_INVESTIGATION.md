# Qwen 3.5 9B Investigation Notes

**Date**: 2026-06-27  
**Status**: Model not working with current Candle integration

---

## Issue Summary

Qwen 3.5 9B (Q4_K_M quantized) produces 0% accuracy with current Candle integration:
- **Raw Output**: Empty
- **Parsed Answer**: Empty  
- **Token Count**: 0
- **Inference Time**: ~8-12 seconds (model loads but generates nothing)

## Comparison

| Model | Status | Accuracy | Notes |
|-------|--------|----------|-------|
| Qwen 3 8B (Q4_K_M) | ✅ Working | 100% | Baseline model |
| Qwen 3.5 9B (Q4_K_M) | ❌ Not Working | 0% | No output generated |

## Possible Causes

1. **Tokenizer Incompatibility**: Qwen 3.5 9B may require different tokenizer
2. **Prompt Format**: ChatML format may differ between versions
3. **Candle Support**: Candle may not fully support Qwen 3.5 9B architecture
4. **Quantization Issue**: Q4_K_M quantization may have issues with 9B model
5. **Model Loading**: Model loads but generation fails silently

## Investigation Steps

1. **Check Candle Version**: Verify Candle supports Qwen 3.5 9B
2. **Test with Different Quantization**: Try Q8_0 or Q6_K instead of Q4_K_M
3. **Check Tokenizer**: Verify tokenizer.json is compatible
4. **Test Minimal Example**: Run `minimal_engine.rs` with Qwen 3.5 9B
5. **Check Candle Issues**: Search GitHub for Qwen 3.5 9B support issues

## Current Status

- **Reverted to Qwen 3 8B**: All model paths reverted to working model
- **Qwen 3.5 9B File**: Still available in `models/Qwen3.5-9B-Q4_K_M.gguf` for future investigation
- **No Breaking Changes**: Hybrid architecture implementation unaffected

## Recommendation

Continue using Qwen 3 8B for now. Investigate Qwen 3.5 9B when:
- Candle adds official support
- Tokenizer compatibility is verified
- Time allows for debugging
