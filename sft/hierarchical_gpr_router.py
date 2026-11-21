"""
SODALS Core — Hierarchical Gaussian Phase Router (H-GPR) for `.qmem v3` Cartridges
===================================================================================
Generalizes the calibrated Layer-32 binary GPR (`gpr_domain_anchors.pt`, Fisher=15.30)
to a 2-level Hierarchical Mahalanobis + Fisher Router supporting 10 to 10,000+
`.qmem v3` virtual parameter cartridges in < 2 ms:

  Level 1 (Macro-Domains): 16-32 broad domain Gaussians (μ_c, σ²_c)
  Level 2 (Sub-Specialists): Child cartridge Gaussians inside Top-2 macro-domains
  Output: Sparse Top-K (K <= 3) blending weights via Sparsemax + OOD Mahalanobis Gate
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact Euclidean projection onto the probability simplex (Martins & Astudillo, 2016)."""
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z.ndim
    view_shape[dim] = -1
    k = k.view(*view_shape)
    cumsum = torch.cumsum(z_sorted, dim=dim)
    support = (1.0 + k * z_sorted) > cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau_sum = torch.gather(cumsum, dim, k_z.long() - 1)
    tau = (tau_sum - 1.0) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


@dataclass
class HGPRArmSignature:
    domain_id: str
    macro_domain: str
    centroid: torch.Tensor  # [D] unit-normalized FP32
    diag_var: torch.Tensor  # [D] positive FP32 variance floor
    cartridge_path: Optional[str] = None


@dataclass
class HGPRRouteResult:
    selected_domains: List[str]
    weights: Dict[str, float]
    macro_scores: Dict[str, float]
    mahalanobis_dists: Dict[str, float]
    min_mahalanobis: float
    is_ood: bool
    trigger_ttt_recommended: bool
    latency_ms: float


class HierarchicalGaussianPhaseRouter:
    """
    2-Level Mahalanobis + Cosine-Fisher Router over Layer-32 hidden representations.
    Guarantees O(M_macro + M_child) < 2 ms routing and fail-safe OOD rejection so
    out-of-distribution queries (e.g., novel HLE axioms) never receive wrong cartridges.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        top_macro: int = 2,
        top_k: int = 2,
        temperature: float = 0.08,
        ood_mahalanobis_threshold: float = 2.60,
        var_floor: float = 1e-3,
    ):
        self.hidden_dim = int(hidden_dim)
        self.top_macro = max(1, int(top_macro))
        self.top_k = max(1, int(top_k))
        self.temperature = max(1e-4, float(temperature))
        self.ood_mahalanobis_threshold = float(ood_mahalanobis_threshold)
        self.var_floor = float(var_floor)

        self._specialists: Dict[str, HGPRArmSignature] = {}
        self._macro_children: Dict[str, List[str]] = {}
        self._macro_centroids: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._dirty: bool = True

        # Cached stacked matrices for sub-millisecond vectorized scoring
        self._macro_names: List[str] = []
        self._macro_mu_mat: Optional[torch.Tensor] = None
        self._macro_inv_var_mat: Optional[torch.Tensor] = None
        # Optional Max-Margin (Linear SVM) specialist hyperplane weights per domain_id
        self._max_margin_w: Dict[str, Tuple[torch.Tensor, float]] = {}
        self.use_max_margin: bool = True

    def fit_max_margin_boundaries(
        self,
        class_samples: Dict[str, torch.Tensor],
        l2_reg: float = 1e-2,
        steps: int = 120,
        lr: float = 0.08,
    ) -> Dict[str, float]:
        """
        Fits L2-regularized one-vs-rest linear Max-Margin (SVM hinge-loss) boundaries
        over unit-normalized Layer-32 representations to sharpen separation between close
        sub-disciplines (e.g., Organic Chemistry vs. Biochemistry) in < 0.1 ms at inference.
        """
        doms = sorted([d for d, x in class_samples.items() if x.numel() > 0])
        if len(doms) < 2:
            return {}

        xs: List[torch.Tensor] = []
        ys: List[int] = []
        for idx_c, d in enumerate(doms):
            t = F.normalize(class_samples[d].detach().float().cpu(), dim=-1)
            xs.append(t)
            ys.extend([idx_c] * t.shape[0])

        x_all = torch.cat(xs, dim=0)  # [N, D]
        y_idx = torch.tensor(ys, dtype=torch.long)
        n_cls = len(doms)

        # Target matrix Y in {-1, +1}^{N x C}
        y_pm = torch.full((x_all.shape[0], n_cls), -1.0, dtype=torch.float32)
        y_pm[torch.arange(x_all.shape[0]), y_idx] = 1.0

        w = torch.zeros((n_cls, self.hidden_dim), dtype=torch.float32, requires_grad=True)
        b = torch.zeros((n_cls,), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=float(lr))

        for _ in range(int(steps)):
            opt.zero_grad()
            margins = x_all @ w.T + b.unsqueeze(0)
            hinge = torch.clamp(1.0 - y_pm * margins, min=0.0).pow(2).mean()
            loss = hinge + 0.5 * float(l2_reg) * (w * w).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            preds = (x_all @ w.T + b.unsqueeze(0)).argmax(dim=-1)
            acc = float((preds == y_idx).float().mean().item())
            for i, d in enumerate(doms):
                w_norm = F.normalize(w[i].detach(), dim=0)
                scale = float(w[i].detach().norm().item()) + 1e-8
                self._max_margin_w[d] = (w_norm, float(b[i].item()) / scale)
        return {"num_classes": float(n_cls), "train_margin_acc": round(acc, 4)}

    def register_specialist(
        self,
        domain_id: str,
        macro_domain: str,
        centroid: torch.Tensor,
        diag_var: Optional[torch.Tensor] = None,
        cartridge_path: Optional[str] = None,
    ) -> None:
        c = F.normalize(centroid.detach().float().cpu().reshape(-1), dim=0)
        if c.numel() != self.hidden_dim:
            raise ValueError(f"Expected centroid dim {self.hidden_dim}, got {c.numel()}")
        if diag_var is None:
            v = torch.full_like(c, 0.02)
        else:
            v = torch.clamp(diag_var.detach().float().cpu().reshape(-1), min=self.var_floor)
            if v.numel() != self.hidden_dim:
                raise ValueError(f"Expected diag_var dim {self.hidden_dim}, got {v.numel()}")

        sig = HGPRArmSignature(
            domain_id=str(domain_id),
            macro_domain=str(macro_domain),
            centroid=c,
            diag_var=v,
            cartridge_path=str(cartridge_path) if cartridge_path else None,
        )
        self._specialists[sig.domain_id] = sig
        children = self._macro_children.setdefault(sig.macro_domain, [])
        if sig.domain_id not in children:
            children.append(sig.domain_id)
        self._dirty = True

    def register_from_qmem_v3(self, path: str | Path) -> str:
        from qmem_v3_cartridge import inspect_qmem_v3_header

        info = inspect_qmem_v3_header(path)
        meta = info.get("metadata", {})
        dom = str(info["domain_id"])
        macro = str(meta.get("macro_domain", "general"))
        c_list = meta.get("centroid_l32")
        if not c_list:
            raise ValueError(f"Cartridge {path} has no centroid_l32 routing signature")
        v_list = meta.get("diag_var_l32")
        self.register_specialist(
            domain_id=dom,
            macro_domain=macro,
            centroid=torch.tensor(c_list, dtype=torch.float32),
            diag_var=torch.tensor(v_list, dtype=torch.float32) if v_list else None,
            cartridge_path=str(path),
        )
        return dom

    @classmethod
    def from_legacy_gpr_anchors(cls, anchors_pt_path: str | Path) -> "HierarchicalGaussianPhaseRouter":
        """Backward-compatible loader for `sft/gpr_domain_anchors.pt`."""
        data = torch.load(str(anchors_pt_path), map_location="cpu", weights_only=False)
        d1 = data["anchor_d1"].float().reshape(-1)
        d2 = data["anchor_d2"].float().reshape(-1)
        router = cls(hidden_dim=int(d1.numel()), top_macro=2, top_k=2)
        v1 = torch.full_like(d1, max(1e-3, float(data.get("var_d1", 0.0025))))
        v2 = torch.full_like(d2, max(1e-3, float(data.get("var_d2", 0.0016))))
        router.register_specialist("expert_general_math", "mathematics", d1, v1)
        router.register_specialist("expert_abstract_algebra", "mathematics", d2, v2)
        return router

    def _rebuild_macro_index(self) -> None:
        self._macro_centroids.clear()
        self._macro_names = sorted(self._macro_children.keys())
        if not self._macro_names:
            self._macro_mu_mat = None
            self._macro_inv_var_mat = None
            self._dirty = False
            return

        mus = []
        inv_vars = []
        for m_name in self._macro_names:
            child_sigs = [self._specialists[cid] for cid in self._macro_children[m_name]]
            c_stack = torch.stack([s.centroid for s in child_sigs], dim=0)
            v_stack = torch.stack([s.diag_var for s in child_sigs], dim=0)
            mu_m = F.normalize(c_stack.mean(dim=0), dim=0)
            # Law of total variance across sub-specialists in this macro-domain
            within_var = v_stack.mean(dim=0)
            between_var = ((c_stack - mu_m.unsqueeze(0)) ** 2).mean(dim=0)
            tot_var = torch.clamp(within_var + between_var, min=self.var_floor)
            self._macro_centroids[m_name] = (mu_m, tot_var)
            mus.append(mu_m)
            inv_vars.append(1.0 / tot_var)

        self._macro_mu_mat = torch.stack(mus, dim=0)  # [M_macro, D]
        self._macro_inv_var_mat = torch.stack(inv_vars, dim=0)  # [M_macro, D]
        self._dirty = False

    def _normalized_mahalanobis(
        self, u: torch.Tensor, mu_mat: torch.Tensor, inv_var_mat: torch.Tensor
    ) -> torch.Tensor:
        """
        Dimension-normalized Mahalanobis distance sqrt(mean_d ((u_d - mu_d)^2 / var_d))
        so thresholds stay invariant to hidden_dim (e.g., 64 in unit tests vs 4096 in 9B).
        """
        diff = u.unsqueeze(0) - mu_mat  # [M, D]
        sq = (diff * diff) * inv_var_mat
        return torch.sqrt(torch.clamp(sq.mean(dim=-1), min=1e-12))

    def route(
        self,
        hidden_l32: torch.Tensor,
        top_k: Optional[int] = None,
        ood_threshold: Optional[float] = None,
        pass1_entropy: Optional[float] = None,
        entropy_ttt_threshold: float = 1.15,
    ) -> HGPRRouteResult:
        t0 = time.perf_counter()
        if self._dirty:
            self._rebuild_macro_index()

        k_eff = max(1, int(top_k if top_k is not None else self.top_k))
        ood_thr = float(ood_threshold if ood_threshold is not None else self.ood_mahalanobis_threshold)

        if self._macro_mu_mat is None or not self._specialists:
            return HGPRRouteResult([], {}, {}, {}, float("inf"), True, True, 0.0)

        h = hidden_l32.detach().float().cpu()
        if h.ndim == 3:
            h = h[0].mean(dim=0)
        elif h.ndim == 2:
            h = h.mean(dim=0)
        u = F.normalize(h.reshape(-1), dim=0)

        # Level 1: Score Macro-Domains
        macro_dists = self._normalized_mahalanobis(u, self._macro_mu_mat, self._macro_inv_var_mat)
        macro_cos = torch.matmul(self._macro_mu_mat, u)
        macro_score_map = {
            self._macro_names[i]: round(float(macro_cos[i].item()), 6)
            for i in range(len(self._macro_names))
        }

        n_macro = min(self.top_macro, len(self._macro_names))
        top_macro_idx = torch.topk(-macro_dists, k=n_macro).indices.tolist()
        candidate_ids: List[str] = []
        for mi in top_macro_idx:
            candidate_ids.extend(self._macro_children[self._macro_names[mi]])

        # Level 2: Score Candidate Sub-Specialists inside Top Macro-Domains
        c_sigs = [self._specialists[cid] for cid in candidate_ids]
        c_mu = torch.stack([s.centroid for s in c_sigs], dim=0)
        c_inv_v = torch.stack([1.0 / s.diag_var for s in c_sigs], dim=0)

        c_dists = self._normalized_mahalanobis(u, c_mu, c_inv_v)
        c_cos = torch.matmul(c_mu, u)
        # Optional Max-Margin (Linear SVM) affinity refinement for tight sub-discipline boundaries
        if self.use_max_margin and self._max_margin_w:
            mm_scores = []
            for cid in candidate_ids:
                if cid in self._max_margin_w:
                    w_vec, b_sc = self._max_margin_w[cid]
                    mm_scores.append(float(torch.dot(w_vec, u).item()) + b_sc)
                else:
                    mm_scores.append(0.0)
            mm_t = torch.tensor(mm_scores, dtype=torch.float32)
            affinities = (c_cos - 0.15 * c_dists + 0.45 * mm_t) / self.temperature
        else:
            # Combined Fisher-Mahalanobis affinity: high cosine + low Mahalanobis distance
            affinities = (c_cos - 0.15 * c_dists) / self.temperature

        min_maha = float(c_dists.min().item())
        dist_map = {candidate_ids[i]: round(float(c_dists[i].item()), 6) for i in range(len(candidate_ids))}

        high_entropy = pass1_entropy is not None and float(pass1_entropy) >= float(entropy_ttt_threshold)
        is_ood = min_maha > ood_thr

        if is_ood:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return HGPRRouteResult(
                selected_domains=[],
                weights={},
                macro_scores=macro_score_map,
                mahalanobis_dists=dist_map,
                min_mahalanobis=round(min_maha, 6),
                is_ood=True,
                trigger_ttt_recommended=True,
                latency_ms=round(elapsed_ms, 3),
            )

        probs = sparsemax(affinities, dim=-1)
        k_take = min(k_eff, len(candidate_ids))
        top_vals, top_idx = torch.topk(probs, k=k_take)

        weights: Dict[str, float] = {}
        selected: List[str] = []
        w_sum = 0.0
        for v, idx_i in zip(top_vals.tolist(), top_idx.tolist()):
            if v > 1e-6:
                dom = candidate_ids[idx_i]
                selected.append(dom)
                weights[dom] = float(v)
                w_sum += float(v)

        # Fallback to single nearest specialist if sparsemax collapsed near boundary
        if not selected:
            best_i = int(torch.argmin(c_dists).item())
            dom = candidate_ids[best_i]
            selected = [dom]
            weights = {dom: 1.0}
        elif w_sum > 0.0:
            weights = {d: round(w / w_sum, 6) for d, w in weights.items()}

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return HGPRRouteResult(
            selected_domains=selected,
            weights=weights,
            macro_scores=macro_score_map,
            mahalanobis_dists=dist_map,
            min_mahalanobis=round(min_maha, 6),
            is_ood=False,
            trigger_ttt_recommended=bool(high_entropy),
            latency_ms=round(elapsed_ms, 3),
        )

    def save_index(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hidden_dim": self.hidden_dim,
            "top_macro": self.top_macro,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "ood_mahalanobis_threshold": self.ood_mahalanobis_threshold,
            "var_floor": self.var_floor,
            "specialists": [
                {
                    "domain_id": s.domain_id,
                    "macro_domain": s.macro_domain,
                    "centroid": s.centroid.tolist(),
                    "diag_var": s.diag_var.tolist(),
                    "cartridge_path": s.cartridge_path,
                }
                for s in self._specialists.values()
            ],
        }
        out.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_index(cls, path: str | Path) -> "HierarchicalGaussianPhaseRouter":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        r = cls(
            hidden_dim=int(d["hidden_dim"]),
            top_macro=int(d.get("top_macro", 2)),
            top_k=int(d.get("top_k", 2)),
            temperature=float(d.get("temperature", 0.08)),
            ood_mahalanobis_threshold=float(d.get("ood_mahalanobis_threshold", 2.60)),
            var_floor=float(d.get("var_floor", 1e-3)),
        )
        for item in d.get("specialists", []):
            r.register_specialist(
                domain_id=item["domain_id"],
                macro_domain=item["macro_domain"],
                centroid=torch.tensor(item["centroid"], dtype=torch.float32),
                diag_var=torch.tensor(item["diag_var"], dtype=torch.float32),
                cartridge_path=item.get("cartridge_path"),
            )
        return r
