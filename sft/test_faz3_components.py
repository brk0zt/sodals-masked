"""
SODALS v2.1 — Verification of Faz 3 Modules
===========================================
Validates:
  1. DifferentiableGPRGate autograd backpropagation (gradients flow to anchors and tau).
  2. AdaptivePonderingODEModule dynamic step allocation and RK4 gradient backpropagation.
"""

import torch
from gpr_differentiable_gate import DifferentiableGPRGate
from adaptive_pondering_ode import AdaptivePonderingODEModule

def test_differentiable_gate():
    print("Testing DifferentiableGPRGate...", flush=True)
    hidden_dim = 4096
    gate = DifferentiableGPRGate(hidden_dim=hidden_dim, temperature=0.03)

    # Mock batch: 2 samples, seq_len=16
    h_mock = torch.randn(2, 16, hidden_dim, requires_grad=True)
    mask_mock = torch.ones(2, 16)
    target_domains = torch.tensor([0.0, 1.0])  # Sample 0: General Math, Sample 1: Abstract Algebra

    # Test 3-tuple default unpacking
    alpha3, z_i3, logits3 = gate(h_mock, mask_mock)
    assert logits3 is not None, "scaled_logits should not be None"

    # Test 2-tuple unpacking
    alpha, z_i = gate(h_mock, mask_mock, return_scaled_logits=False)
    print(f"  alpha: {alpha.detach().tolist()}")
    print(f"  z_i  : {z_i.detach().tolist()}")

    loss = gate.compute_gate_loss(alpha, target_domains)
    loss.backward()

    assert gate.anchor_d1.grad is not None, "anchor_d1 received no gradient!"
    assert gate.anchor_d2.grad is not None, "anchor_d2 received no gradient!"
    assert gate.tau.grad is not None, "tau received no gradient!"
    assert h_mock.grad is not None, "hidden_states received no gradient!"

    print(f"  Gate Loss: {loss.item():.4f}")
    print(f"  anchor_d1 grad norm: {gate.anchor_d1.grad.norm().item():.6f}")
    print(f"  anchor_d2 grad norm: {gate.anchor_d2.grad.norm().item():.6f}")
    print(f"  tau grad           : {gate.tau.grad.item():.6f}")
    print("  [PASS] DifferentiableGPRGate autograd verified successfully!", flush=True)

def test_adaptive_pondering():
    print("\nTesting AdaptivePonderingODEModule...", flush=True)
    hidden_dim = 4096
    ode = AdaptivePonderingODEModule(hidden_dim=hidden_dim, min_steps=2, max_steps=5, base_lambda=0.3)

    h0_math = torch.randn(1, 8, hidden_dim, requires_grad=True)
    h_out_math, steps_math, lambda_math = ode(h0_math, alpha=0.1)
    print(f"  General Math (alpha=0.1) -> Steps: {steps_math} (expected: 2), Lambda: {lambda_math:.3f}")
    assert steps_math == 2, f"Expected 2 steps, got {steps_math}"

    h0_alg = torch.randn(1, 8, hidden_dim, requires_grad=True)
    h_out_alg, steps_alg, lambda_alg = ode(h0_alg, alpha=0.95)
    print(f"  Abstract Algebra (alpha=0.95) -> Steps: {steps_alg} (expected: 5), Lambda: {lambda_alg:.3f}")
    assert steps_alg == 5, f"Expected 5 steps, got {steps_alg}"

    # Gradient check through RK4
    loss_ode = h_out_alg.sum()
    loss_ode.backward()
    assert ode.arkhe_proj.weight.grad is not None, "arkhe_proj received no gradient!"
    assert h0_alg.grad is not None, "h0 received no gradient!"
    print(f"  RK4 flow grad norm: {ode.arkhe_proj.weight.grad.norm().item():.6f}")
    print("  [PASS] AdaptivePonderingODEModule dynamic steps and backprop verified successfully!", flush=True)

if __name__ == "__main__":
    test_differentiable_gate()
    test_adaptive_pondering()
    print("\nALL FAZ 3 MODULE TESTS COMPLETED SUCCESSFULLY!")
