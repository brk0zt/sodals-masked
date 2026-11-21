import os
import sys
import tempfile
import unittest
from pathlib import Path

SFT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32" and sys.version_info >= (3, 13):
    import smart_app_control_compat  # noqa: F401

import torch
import torch.nn.functional as F

from eval_qmem_v3_pilot_benchmark import compute_paired_summary, exact_mcnemar_two_sided, grade_pilot_answer
from hierarchical_gpr_router import HierarchicalGaussianPhaseRouter
from qmem_apex_bandit_search import (
    CartridgeDiverseAllocator,
    CdclConflictMemory,
    ContextualCartridgeBandit,
)
from qmem_v3_cartridge import QmemV3Cartridge, QmemV3Metadata, apply_cartridge_bank3_to_deltanet_layers
from svon_cvnn_deltanet import HolomorphicCVNNDeltaNetWrapper
from svon_kuramoto_equilibrium import upgrade_model_to_svon_cgw
from svon_looped_depth_ponder import ZeroInitLoopedDepthWrapper


def _build_tiny_qwen35():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.25,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    torch.manual_seed(123)
    return Qwen3_5ForCausalLM(cfg).eval()


class TestSodalsApexV35(unittest.TestCase):
    def test_contextual_bandit_ucb1_thompson_and_inherit_stats(self):
        for strat in ("ucb1", "thompson"):
            bandit = ContextualCartridgeBandit(strategy=strat, exploration_c=0.5, bandit_weight=1.0, seed=7)
            for _ in range(12):
                bandit.observe_reward("phys_qft", +1.0, macro_domain="physics")
                bandit.observe_reward("phys_bad", -1.0, macro_domain="physics")

            ranked = bandit.rank_candidates({"phys_qft": 0.5, "phys_bad": 0.5}, macro_domain="physics")
            self.assertEqual(ranked[0][0], "phys_qft")

            # TTT cartridge inherits parent stats cleanly without cold-start reset
            child = bandit.inherit_stats_from("ttt_phys_1", "phys_qft", macro_domain="physics", transfer_ratio=0.5)
            self.assertGreaterEqual(child.pulls, 1)
            self.assertGreater(child.mean_reward, 0.5)

            with tempfile.TemporaryDirectory() as td:
                fp = Path(td) / f"bandit_{strat}.json"
                bandit.save(fp)
                reloaded = ContextualCartridgeBandit.load(fp)
                self.assertEqual(reloaded._arms[("physics", "phys_qft")].pulls, 12)

    def test_cdcl_conflict_memory_and_cartridge_diverse_successive_halving(self):
        bandit = ContextualCartridgeBandit(strategy="ucb1", enabled=True)
        cdcl = CdclConflictMemory(block_threshold=1, enabled=True)
        alloc = CartridgeDiverseAllocator(bandit=bandit, cdcl=cdcl, initial_g=4, max_g=16, agreement_stop_threshold=0.60)

        # Block failing arm "chem_bad" on problem q_42
        cdcl.record_conflict("q_42", "chem_bad", reason="sympy_contradiction")
        self.assertTrue(cdcl.is_blocked("q_42", "chem_bad"))

        plans = alloc.plan_stage_branches(
            problem_sig="q_42",
            macro_domain="chemistry",
            hgpr_weights={"chem_bad": 0.95, "chem_org": 0.65, "chem_phys": 0.35},
            ttt_domain_id="ttt_q42",
        )
        self.assertEqual(len(plans), 4)
        # Blocked arm must NEVER appear in any planned branch
        for bp in plans:
            self.assertNotIn("chem_bad", bp.cartridge_weights)
        # Branches must span distinct cartridge configs (decorrelating errors!)
        unique_labels = {bp.arm_label for bp in plans}
        self.assertGreaterEqual(len(unique_labels), 3)

        # Successive halving: high agreement (0.75) + no contradiction -> stop early
        esc_hi, _ = alloc.should_escalate(4, max_agreement_ratio=0.75, has_symbolic_contradiction=False)
        self.assertFalse(esc_hi)
        # Low agreement (0.50) or contradiction -> escalate G=4 -> +4 (G=8)
        esc_lo, add_g = alloc.should_escalate(4, max_agreement_ratio=0.50, has_symbolic_contradiction=False)
        self.assertTrue(esc_lo)
        self.assertEqual(add_g, 4)

    def test_phase_multiplexed_bank3_and_destructive_pi_erasure(self):
        # 1. Verify destructive Δφ = π erasure removes falsified branch projection cleanly
        k = F.normalize(torch.randn(1, 4, 16), dim=-1)
        v_bad = torch.randn(1, 4, 16)
        k_good = F.normalize(torch.randn(1, 4, 16), dim=-1)
        # Orthogonalize k_good to k
        k_good = F.normalize(k_good - (k_good * k).sum(dim=-1, keepdim=True) * k, dim=-1)
        v_good = torch.randn(1, 4, 16)

        state = k.unsqueeze(-1) * v_bad.unsqueeze(-2) + k_good.unsqueeze(-1) * v_good.unsqueeze(-2)
        erased = HolomorphicCVNNDeltaNetWrapper.erase_branch_from_state(state, k, v_bad, beta=1.0)

        read_bad = (erased * k.unsqueeze(-1)).sum(dim=-2)
        read_good = (erased * k_good.unsqueeze(-1)).sum(dim=-2)
        self.assertLess(float(read_bad.abs().max().item()), 1e-5)
        self.assertTrue(torch.allclose(read_good, v_good, atol=1e-5))

        # 2. Verify Phase-Multiplexed Bank 3 on real tiny Qwen3_5 with 2 active cartridges
        model = _build_tiny_qwen35()
        upgrade_model_to_svon_cgw(model, phase_rank=8)
        c1 = QmemV3Cartridge(
            metadata=QmemV3Metadata(domain_id="c1"),
            tensors={
                "bank3.layers.0.k_cart": torch.randn(2, 4, 16),
                "bank3.layers.0.v_cart": torch.randn(2, 4, 16),
                "bank3.layers.0.beta_cart": torch.zeros(2, 4),
            },
            bank_map={"bank3.layers.0.k_cart": 3, "bank3.layers.0.v_cart": 3, "bank3.layers.0.beta_cart": 3},
            layer_map={"bank3.layers.0.k_cart": 0, "bank3.layers.0.v_cart": 0, "bank3.layers.0.beta_cart": 0},
        )
        c2 = QmemV3Cartridge(
            metadata=QmemV3Metadata(domain_id="c2"),
            tensors={
                "bank3.layers.0.k_cart": torch.randn(2, 4, 16),
                "bank3.layers.0.v_cart": torch.randn(2, 4, 16),
                "bank3.layers.0.beta_cart": torch.zeros(2, 4),
            },
            bank_map={"bank3.layers.0.k_cart": 3, "bank3.layers.0.v_cart": 3, "bank3.layers.0.beta_cart": 3},
            layer_map={"bank3.layers.0.k_cart": 0, "bank3.layers.0.v_cart": 0, "bank3.layers.0.beta_cart": 0},
        )
        x = torch.randint(1, 200, (1, 8))
        with torch.no_grad():
            y_base = model(input_ids=x, use_cache=False).logits
        apply_cartridge_bank3_to_deltanet_layers(model, [(c1, 0.6), (c2, 0.4)])
        with torch.no_grad():
            y_zero = model(input_ids=x, use_cache=False).logits
        self.assertLess(float((y_zero - y_base).abs().max().item()), 1e-5)

    def test_zero_init_looped_depth_exact_submanifold_and_gradient_flow(self):
        # Simple residual block emulating top decoder layers
        block = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Tanh())
        wrapper = ZeroInitLoopedDepthWrapper([block], max_extra_loops=2, entropy_threshold=0.80, enabled=True)
        h = torch.randn(1, 6, 64)

        # 1. At init (gamma_loop == 0.0), eval output MUST be bit-identical (< 1e-6)
        wrapper.eval()
        y_init, tel_init = wrapper(h, step_entropy=1.5)
        self.assertFalse(tel_init.triggered)
        self.assertLess(float((y_init - h).abs().max().item()), 1e-6)

        # 2. Low entropy (< 0.80) skips loops even when gamma_loop != 0
        with torch.no_grad():
            wrapper.gamma_loop.fill_(0.25)
        y_low, tel_low = wrapper(h, step_entropy=0.20)
        self.assertFalse(tel_low.triggered)
        self.assertLess(float((y_low - h).abs().max().item()), 1e-6)

        # 3. High entropy + non-zero gamma executes loops and flows finite gradients to gamma_loop
        wrapper.train()
        y_hi, tel_hi = wrapper(h, step_entropy=1.40)
        self.assertTrue(tel_hi.triggered)
        self.assertEqual(tel_hi.loops_executed, 2)
        y_hi.sum().backward()
        self.assertIsNotNone(wrapper.gamma_loop.grad)
        self.assertTrue(torch.isfinite(wrapper.gamma_loop.grad).all())
        self.assertGreater(float(wrapper.gamma_loop.grad.abs().item()), 0.0)

    def test_max_margin_hgpr_sharpens_close_subdisciplines(self):
        torch.manual_seed(99)
        router = HierarchicalGaussianPhaseRouter(hidden_dim=32, top_macro=2, top_k=1, ood_mahalanobis_threshold=5.0)
        shared = F.normalize(torch.randn(32), dim=0)
        delta = F.normalize(torch.randn(32), dim=0)
        delta = F.normalize(delta - torch.dot(delta, shared) * shared, dim=0)

        s_org = F.normalize(shared.unsqueeze(0) + 0.12 * delta.unsqueeze(0) + 0.03 * torch.randn(20, 32), dim=-1)
        s_bio = F.normalize(shared.unsqueeze(0) - 0.12 * delta.unsqueeze(0) + 0.03 * torch.randn(20, 32), dim=-1)

        router.register_specialist("chem_org", "chem", s_org.mean(dim=0), torch.full((32,), 0.01))
        router.register_specialist("chem_bio", "chem", s_bio.mean(dim=0), torch.full((32,), 0.01))

        stats = router.fit_max_margin_boundaries({"chem_org": s_org, "chem_bio": s_bio}, steps=80)
        self.assertGreaterEqual(stats["train_margin_acc"], 0.95)

        q_org = F.normalize(shared + 0.10 * delta, dim=0)
        q_bio = F.normalize(shared - 0.10 * delta, dim=0)
        self.assertEqual(router.route(q_org, top_k=1).selected_domains[0], "chem_org")
        self.assertEqual(router.route(q_bio, top_k=1).selected_domains[0], "chem_bio")

    def test_pilot_grader_and_exact_mcnemar_math(self):
        ok1, p1 = grade_pilot_answer("Step 1... Therefore the answer is (D)", "D", task_type="mcq")
        self.assertTrue(ok1)
        self.assertEqual(p1, "D")
        self.assertAlmostEqual(exact_mcnemar_two_sided(6, 0), 0.03125, places=5)
        summ = compute_paired_summary([False, False, True, False], [True, True, True, False])
        self.assertEqual(summ["b01_gains"], 2)
        self.assertEqual(summ["b10_losses"], 0)


if __name__ == "__main__":
    unittest.main()
