"""
GPU-Free Mathematical & Architectural Unit Tests for Epoch III (`Svon-CGW`)
===========================================================================
Verifies:
  1. Exact Submanifold Theorem (CVNN-DeltaNet): At initialization (omega_t = 0, alpha_im = 0),
     `HolomorphicCVNNDeltaNetWrapper` and `recurrent_complex_gated_delta_rule` match
     Qwen 3.5's native `Qwen3_5GatedDeltaNet` (`torch_recurrent_gated_delta_rule` &
     `torch_chunk_gated_delta_rule`) to < 1e-6 (bit-exact at t=0).
  2. Constructive vs. Destructive Complex Wave Interference:
     Writing a key-value association at phase 0 and a second update at phase pi (e^{-i*pi} = -1)
     destructively cancels the stored state, while phase 0 constructively reinforces it.
  3. 2-Hop Relational Sheaf-GNN Transitivity & 192 Position-Free Channel Isolation:
     - At alpha_hop = 0, matches 1-hop `P @ V` bit-identically.
     - At alpha_hop > 0, propagates 2-hop transitive dependency (`w -> v -> u`) in a single
       layer across dims `[64:256]` while leaving 1D RoPE channels `[0:64]` untouched.
  4. Kuramoto Phase Synchronization & Full Hybrid Decoder Stack Upgrade + Rust `.qmem` Export:
     - `KuramotoPhaseEquilibrium` monotonically increases the phase order parameter `R_order`.
     - `upgrade_model_to_svon_cgw` upgrades a hybrid Decoder stack, pipes CVNN phase to
       Sheaf-GNN, propagates finite non-zero gradients to CVNN+GNN parameters, and exports
       valid 32-byte `QMEM` header + `QmemManifest`.
"""

import json
import os
import struct
import sys
import tempfile
import unittest

SFT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32" and sys.version_info >= (3, 13):
    import smart_app_control_compat  # noqa: F401

import math
import torch
import torch.nn as nn

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextModel,
    torch_recurrent_gated_delta_rule,
)

from svon_cvnn_deltanet import (
    HolomorphicCVNNDeltaNetWrapper,
    apply_complex_phase_rotation,
    recurrent_complex_gated_delta_rule,
)
from svon_sheaf_gnn_attention import (
    RelationalSheafGNNAttention,
    compute_phase_coherence_matrix,
    two_hop_sheaf_message_passing,
)
from svon_kuramoto_equilibrium import (
    KuramotoPhaseEquilibrium,
    collect_svon_cgw_state_dict,
    compute_kuramoto_order_parameter,
    export_svon_cgw_to_rust_qmem,
    load_svon_cgw_state_dict,
    upgrade_model_to_svon_cgw,
)
from train_svon_cgw_architecture import compute_phase_contrastive_loss


class TestSvonCGWHybridGnnCvnn(unittest.TestCase):
    def _make_mini_config(self) -> Qwen3_5TextConfig:
        return Qwen3_5TextConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            partial_rotary_factor=0.25,
            pad_token_id=0,
        )

    def test_exact_submanifold_theorem_cvnn_matches_native_qwen35_at_zero_phase(self):
        torch.manual_seed(42)
        cfg = self._make_mini_config()
        base_dn = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).eval()
        x = torch.randn(2, 12, cfg.hidden_size)

        with torch.no_grad():
            native_out = base_dn(x)

        cvnn_dn = HolomorphicCVNNDeltaNetWrapper(base_dn, phase_rank=8).eval()
        with torch.no_grad():
            cvnn_out = cvnn_dn(x)

        max_diff = float((native_out - cvnn_out).abs().max().item())
        self.assertEqual(max_diff, 0.0, f"Expected exact 0.0 submanifold identity, got {max_diff}")

        # Also verify standalone recurrent_complex_gated_delta_rule at omega=0, alpha_im=0
        q = torch.randn(2, 10, 2, 32)
        k = torch.randn(2, 10, 2, 32)
        v = torch.randn(2, 10, 4, 32)
        g = -torch.rand(2, 10, 4) * 0.5
        beta = torch.sigmoid(torch.randn(2, 10, 4))

        ref_out, ref_state = torch_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        c_out, c_state, _ = recurrent_complex_gated_delta_rule(
            q, k, v, g=g, beta=beta, omega=None, alpha_im=None, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        self.assertLess(float((ref_out - c_out).abs().max().item()), 1e-6)
        self.assertLess(float((ref_state - c_state).abs().max().item()), 1e-6)

    def test_constructive_and_destructive_complex_wave_interference(self):
        # Two-step memory test: write vector v at step 0 with key k0, then write -v or +v via phase shift pi at step 1
        bsz, seq_len, h_k, d_k = 1, 2, 1, 8
        h_v, d_v = 1, 8
        k_vec = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).view(1, 1, 1, 8)
        k = k_vec.repeat(1, 2, 1, 1)
        q = k.clone()
        v = torch.ones((1, 2, 1, 8))
        g = torch.zeros((1, 2, 1))          # zero decay (pure unitary storage)
        beta = torch.full((1, 2, 1), 0.5)   # equal write weight

        # Case A: Constructive interference (omega = [0, 0] -> Delta Theta = 0)
        omega_constructive = torch.zeros((1, 2, 1, 4))
        out_con, state_con, _ = recurrent_complex_gated_delta_rule(
            q, k, v, g=g, beta=beta, omega=omega_constructive, output_final_state=True
        )

        # Case B: Destructive interference on step 2 (omega = [0, pi] -> e^{-i*pi} = -1)
        omega_destructive = torch.zeros((1, 2, 1, 4))
        omega_destructive[:, 1, :, :] = math.pi
        out_des, state_des, _ = recurrent_complex_gated_delta_rule(
            q, k, v, g=g, beta=beta, omega=omega_destructive, output_final_state=True
        )

        norm_con = float(state_con.norm().item())
        norm_des = float(state_des.norm().item())
        # Constructive interference builds strong associative state; phase-pi opposition cancels the write!
        self.assertGreater(norm_con, norm_des * 2.0)

    def test_two_hop_sheaf_gnn_transitivity_and_rope_channel_isolation(self):
        # 3 nodes: 0 -> 1 -> 2. Direct 1-hop attention from node 2 only sees node 1 (P[2, 0] == 0).
        # With 2-hop Sheaf-GNN diffusion (P_graph @ P), node 2 receives node 0's feature in 1 layer!
        bsz, heads, seq_len, head_dim = 1, 2, 3, 16
        ro_free_start = 4  # first 4 dims are RoPE; remaining 12 dims are position-free graph channels

        p_1hop = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # node 0 attends to 0
                [0.8, 0.2, 0.0],  # node 1 attends strongly to node 0
                [0.0, 1.0, 0.0],  # node 2 attends ONLY to node 1 (zero 1-hop weight on node 0!)
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3).repeat(1, heads, 1, 1)

        values = torch.zeros((1, heads, 3, head_dim), dtype=torch.float32)
        values[:, :, 0, :] = 10.0  # signal lives ONLY at node 0!

        # 1. At alpha_hop = 0, matches 1-hop (node 2 gets 0.0 from node 0!)
        alpha_zero = torch.zeros(heads)
        out_1hop = two_hop_sheaf_message_passing(p_1hop, values, alpha_zero, ro_free_start_dim=ro_free_start)
        self.assertEqual(float(out_1hop[:, :, 2, :].abs().max().item()), 0.0)

        # 2. At alpha_hop > 0, node 2 receives node 0's signal via 2-hop path (2 -> 1 -> 0) on [ro_free_start:]
        alpha_pos = torch.full((heads,), 0.8)
        out_2hop = two_hop_sheaf_message_passing(p_1hop, values, alpha_pos, ro_free_start_dim=ro_free_start)
        # RoPE channels [0:4] at node 2 stay strictly untouched (0.0)
        self.assertEqual(float(out_2hop[:, :, 2, :ro_free_start].abs().max().item()), 0.0)
        # Position-free graph channels [4:16] at node 2 receive strong 2-hop message (> 5.0)!
        self.assertGreater(float(out_2hop[:, :, 2, ro_free_start:].mean().item()), 5.0)

    def test_kuramoto_synchronization_and_full_model_upgrade_with_qmem_export(self):
        torch.manual_seed(99)
        # 1. Kuramoto phase locking increases R_order
        desync_phase = (torch.rand(2, 8, 2, 16) - 0.5) * 3.0
        eq = KuramotoPhaseEquilibrium(sync_threshold=0.95, coupling_strength=0.45, max_sync_steps=4)
        synced_phase, k_meta = eq(desync_phase)
        self.assertGreater(k_meta["r_final"], k_meta["r_initial"])

        # 2. Upgrade full mini Qwen3_5TextModel (3 linear_attn + 1 full_attention)
        cfg = self._make_mini_config()
        model = Qwen3_5TextModel(cfg).eval()
        ids = torch.randint(1, 200, (1, 10))

        with torch.no_grad():
            orig_hidden = model(input_ids=ids).last_hidden_state

        up_stats = upgrade_model_to_svon_cgw(model, phase_rank=4, ro_free_start_dim=8, freeze_base=True)
        self.assertEqual(up_stats["cvnn_layer_indices"], [0, 1, 2])
        self.assertEqual(up_stats["sheaf_gnn_layer_indices"], [3])

        # Verify Exact Submanifold Theorem across the ENTIRE 4-layer hybrid model in eval() at t=0
        model.eval()
        with torch.no_grad():
            cgw_hidden = model(input_ids=ids).last_hidden_state
        self.assertEqual(float((orig_hidden - cgw_hidden).abs().max().item()), 0.0)

        # 2b. Verify NON-ZERO first-step gradient flow from STRICT ZERO-INIT when wrappers are in train mode
        for m in model.modules():
            if m.__class__.__name__ in ("HolomorphicCVNNDeltaNetWrapper", "RelationalSheafGNNAttention"):
                m.train()
        out0 = model(input_ids=ids).last_hidden_state
        # At strict zero-init forward STILL matches native Qwen within float tolerances
        self.assertLess(float((orig_hidden - out0.detach()).abs().max().item()), 1e-5)
        out0.pow(2).mean().backward()
        self.assertGreater(float(model.layers[2].linear_attn.phase_proj_up.weight.grad.abs().max().item()), 0.0)
        self.assertGreater(float(model.layers[2].linear_attn.alpha_im.grad.abs().max().item()), 0.0)
        self.assertGreater(float(model.layers[3].self_attn.alpha_hop.grad.abs().max().item()), 0.0)
        self.assertGreater(float(model.layers[3].self_attn.gamma_phase.grad.abs().max().item()), 0.0)
        model.zero_grad(set_to_none=True)

        # 3. Verify gradient flow through both CVNN (`phase_proj_up`, `alpha_im`) and Sheaf-GNN (`alpha_hop`, `gamma_phase`)
        # with explicit phase perturbation + phase-contrastive loss

        out = model(input_ids=ids).last_hidden_state
        phases = [model.layers[i].linear_attn.compute_omega(out) for i in [0, 1, 2]]
        p_loss, _ = compute_phase_contrastive_loss(phases, [p + math.pi * 0.5 for p in phases])
        total_loss = out.pow(2).mean() + p_loss
        total_loss.backward()

        self.assertIsNotNone(model.layers[2].linear_attn.phase_proj_up.weight.grad)
        self.assertGreater(float(model.layers[2].linear_attn.phase_proj_up.weight.grad.abs().max().item()), 0.0)
        self.assertIsNotNone(model.layers[3].self_attn.alpha_hop.grad)
        self.assertGreater(float(model.layers[3].self_attn.alpha_hop.grad.abs().max().item()), 0.0)
        self.assertIsNotNone(model.layers[3].self_attn.gamma_phase.grad)
        self.assertGreater(float(model.layers[3].self_attn.gamma_phase.grad.abs().max().item()), 0.0)

        # 4. Verify Rust `.qmem` (32-byte QmemHeader) + `.safetensors` + `QmemManifest` export
        state = collect_svon_cgw_state_dict(model)
        with tempfile.TemporaryDirectory() as tmpd:
            exp = export_svon_cgw_to_rust_qmem(state, out_dir=tmpd, basename="test_cgw")
            self.assertTrue(os.path.exists(exp["safetensors_path"]))
            self.assertTrue(os.path.exists(exp["qmem_path"]))
            self.assertTrue(os.path.exists(exp["manifest_path"]))

            with open(exp["qmem_path"], "rb") as qf:
                hdr = qf.read(32)
            magic, ver_maj, ver_min, dtype_code, _pad, length, srate, _ts, label = struct.unpack("<4sBBBBIfQ8s", hdr)
            self.assertEqual(magic, b"QMEM")
            self.assertEqual((ver_maj, ver_min), (2, 1))
            self.assertEqual(dtype_code, 2)
            self.assertEqual(length, exp["total_floats"])

            with open(exp["manifest_path"], "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
            self.assertEqual(len(manifest["layers"]), 4)

            # Verify round-trip load_svon_cgw_state_dict & live Kuramoto rollback synchronizer
            from svon_kuramoto_equilibrium import synchronize_model_cvnn_phases_on_rollback
            loaded_cnt = load_svon_cgw_state_dict(model, exp["safetensors_path"])
            self.assertEqual(loaded_cnt, len(state))
            rb_meta = synchronize_model_cvnn_phases_on_rollback(model, coupling_strength=0.35, max_sync_steps=3)
            self.assertEqual(rb_meta["synced_cvnn_layers"], 3)
            self.assertGreaterEqual(rb_meta["mean_r_after"], rb_meta["mean_r_before"])

            # 5. Wave 2 checks: manifest byte length, rollback tail update, differentiable Kuramoto K, zero-init kappa_dead
            self.assertEqual(manifest["layers"][0]["length"], manifest["layers"][0]["num_elements"] * 4)
            for l_idx in [0, 1, 2]:
                cw = model.layers[l_idx].linear_attn
                self.assertIsNotNone(cw._cached_phase_tail)
                self.assertTrue(torch.allclose(cw._cached_phase_tail, cw.last_cumulative_phase[:, -1:, :, :]))

            eq_diff = KuramotoPhaseEquilibrium(sync_threshold=0.99, coupling_strength=0.35, max_sync_steps=2)
            out_p, _ = eq_diff(desync_phase)
            out_p.pow(2).mean().backward()
            self.assertIsNotNone(eq_diff.coupling_strength.grad)
            self.assertGreater(float(eq_diff.coupling_strength.grad.abs().item()), 0.0)

            # Verify kappa_dead == 0 produces exact 0.0 dead-mask penalty (softplus(0) - ln(2) == 0)
            gw = model.layers[3].self_attn
            gw.eval()
            gw.kappa_dead.data.zero_()
            gw.alpha_hop.data.zero_()
            gw.gamma_phase.data.zero_()
            gw.eta_sym.data.zero_()
            gw.set_graph_context(dead_mask=torch.ones((1, 1, 10, 10)))
            with torch.no_grad():
                out_with_dead = model(input_ids=ids).last_hidden_state
            gw.set_graph_context(dead_mask=None)
            self.assertLess(float((out_with_dead - orig_hidden).abs().max().item()), 1e-5)


if __name__ == "__main__":
    unittest.main()
