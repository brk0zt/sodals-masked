"""
SODALS-Apex (`.qmem v3.5`) — Contextual MAB, CDCL Conflict Memory & Cartridge-Diverse Search
============================================================================================
Unifies three battle-tested ideas from SODALS's ARC-AGI-3 Rust core (`CartridgeLibrary` UCB1,
`CdclClauseDb`, `ChronosWeaver` virtual reset) with `.qmem v3` virtual parameter cartridges:

  1. `ContextualCartridgeBandit`: UCB1 + Discounted Thompson Sampling over `(macro_domain, domain_id)`
     arms, updated from verified outcomes (SymPy / pytest / self-consistency agreement) rather than
     noisy proxies, with `inherit_stats_from()` so newly promoted TTT-QMEM cartridges avoid cold start.
  2. `CdclConflictMemory`: Stores verified failure clauses `(problem_sig, arm_or_tactic)` to block
     repeated failing cartridges/tactics across G=4..16 rollouts and multi-turn SWE-Pro episodes.
  3. `CartridgeDiverseAllocator`: Successive Halving (`G=4 -> 8 -> 16`) rollout planner that breaks
     correlated greedy hallucinations by spreading branches across orthogonal cartridge slots +
     TTT-QMEM + Base configurations (< 2 ms hot-swap via `QmemV3Manager`).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class BanditArmStats:
    domain_id: str
    macro_domain: str = "general"
    pulls: int = 0
    reward_sum: float = 0.0
    # Beta / Gaussian posterior parameters for Discounted Thompson Sampling
    ts_alpha: float = 1.0
    ts_beta: float = 1.0
    recent_rewards: List[float] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return (self.reward_sum / self.pulls) if self.pulls > 0 else 0.0


class ContextualCartridgeBandit:
    """
    Closed-loop Contextual Multi-Armed Bandit (UCB1 or Discounted Thompson Sampling)
    layered directly on top of `HierarchicalGaussianPhaseRouter` affinities.
    Toggleable (`enabled=False` -> pure H-GPR for strict Claude ablations).
    """

    def __init__(
        self,
        strategy: str = "ucb1",  # "ucb1" or "thompson"
        exploration_c: float = 1.15,
        bandit_weight: float = 0.35,
        discount_gamma: float = 0.95,
        recent_window: int = 16,
        enabled: bool = True,
        seed: Optional[int] = 42,
    ):
        self.strategy = "thompson" if str(strategy).lower() == "thompson" else "ucb1"
        self.exploration_c = float(exploration_c)
        self.bandit_weight = float(bandit_weight)
        self.discount_gamma = max(0.5, min(1.0, float(discount_gamma)))
        self.recent_window = max(4, int(recent_window))
        self.enabled = bool(enabled)
        self._rng = random.Random(seed)
        # Keyed by (macro_domain, domain_id)
        self._arms: Dict[Tuple[str, str], BanditArmStats] = {}
        self._macro_pulls: Dict[str, int] = {}

    def register_arm(self, domain_id: str, macro_domain: str = "general") -> BanditArmStats:
        key = (str(macro_domain), str(domain_id))
        if key not in self._arms:
            self._arms[key] = BanditArmStats(domain_id=str(domain_id), macro_domain=str(macro_domain))
        return self._arms[key]

    def inherit_stats_from(
        self,
        new_domain_id: str,
        parent_domain_id: str,
        macro_domain: str = "general",
        transfer_ratio: float = 0.5,
    ) -> BanditArmStats:
        """
        Transfers decayed prior statistics from a parent cartridge to a newly synthesized
        TTT-QMEM cartridge so it enters the library without infinite cold-start bias.
        """
        dst = self.register_arm(new_domain_id, macro_domain)
        src = self._arms.get((str(macro_domain), str(parent_domain_id)))
        if src is not None and src.pulls > 0:
            ratio = max(0.1, min(1.0, float(transfer_ratio)))
            dst.pulls = max(1, int(round(src.pulls * ratio)))
            dst.reward_sum = src.reward_sum * (dst.pulls / float(src.pulls))
            dst.ts_alpha = 1.0 + max(0.0, (src.ts_alpha - 1.0) * ratio)
            dst.ts_beta = 1.0 + max(0.0, (src.ts_beta - 1.0) * ratio)
            dst.recent_rewards = list(src.recent_rewards[-self.recent_window :])
        return dst

    def score_arm(self, domain_id: str, macro_domain: str = "general", hgpr_affinity: float = 0.0) -> float:
        if not self.enabled:
            return float(hgpr_affinity)
        arm = self.register_arm(domain_id, macro_domain)
        total_m = max(1, self._macro_pulls.get(str(macro_domain), 0))
        if self.strategy == "thompson":
            bonus = self._rng.betavariate(max(0.1, arm.ts_alpha), max(0.1, arm.ts_beta))
            # Map [0, 1] Beta sample to centered [-0.5, +0.5] bonus
            bandit_term = bonus - 0.5
        else:
            if arm.pulls == 0:
                # Bounded cold-start optimism so H-GPR still prevents wildly irrelevant domains
                bandit_term = self.exploration_c * 1.25
            else:
                bandit_term = arm.mean_reward + self.exploration_c * math.sqrt(math.log(total_m + 1.0) / arm.pulls)
        return float(hgpr_affinity) + self.bandit_weight * float(bandit_term)

    def rank_candidates(
        self,
        candidates: Dict[str, float],
        macro_domain: str = "general",
        blocked_domains: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        blocked = blocked_domains or set()
        scored: List[Tuple[str, float]] = []
        for dom, aff in candidates.items():
            if dom in blocked:
                continue
            scored.append((dom, self.score_arm(dom, macro_domain=macro_domain, hgpr_affinity=aff)))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored

    def observe_reward(
        self,
        domain_id: str,
        reward: float,
        macro_domain: str = "general",
    ) -> BanditArmStats:
        """
        Records a verified reward in [-1.0, +1.0] (e.g., +1.0 verified pass, 0.0 neutral,
        -0.5/-1.0 symbolic contradiction or test failure).
        """
        arm = self.register_arm(domain_id, macro_domain)
        r = max(-1.0, min(1.0, float(reward)))
        arm.pulls += 1
        arm.reward_sum += r
        arm.recent_rewards.append(r)
        if len(arm.recent_rewards) > self.recent_window:
            arm.recent_rewards = arm.recent_rewards[-self.recent_window :]

        # Discounted Beta update for non-stationary task adaptation
        p01 = 0.5 * (r + 1.0)  # map [-1, 1] -> [0, 1]
        arm.ts_alpha = 1.0 + self.discount_gamma * (arm.ts_alpha - 1.0) + p01
        arm.ts_beta = 1.0 + self.discount_gamma * (arm.ts_beta - 1.0) + (1.0 - p01)

        self._macro_pulls[str(macro_domain)] = self._macro_pulls.get(str(macro_domain), 0) + 1
        return arm

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy": self.strategy,
            "exploration_c": self.exploration_c,
            "bandit_weight": self.bandit_weight,
            "discount_gamma": self.discount_gamma,
            "macro_pulls": self._macro_pulls,
            "arms": [
                {
                    "domain_id": a.domain_id,
                    "macro_domain": a.macro_domain,
                    "pulls": a.pulls,
                    "reward_sum": a.reward_sum,
                    "ts_alpha": a.ts_alpha,
                    "ts_beta": a.ts_beta,
                    "recent_rewards": a.recent_rewards,
                }
                for a in self._arms.values()
            ],
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ContextualCartridgeBandit":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        b = cls(
            strategy=d.get("strategy", "ucb1"),
            exploration_c=float(d.get("exploration_c", 1.15)),
            bandit_weight=float(d.get("bandit_weight", 0.35)),
            discount_gamma=float(d.get("discount_gamma", 0.95)),
        )
        b._macro_pulls = {str(k): int(v) for k, v in d.get("macro_pulls", {}).items()}
        for item in d.get("arms", []):
            arm = b.register_arm(item["domain_id"], item.get("macro_domain", "general"))
            arm.pulls = int(item.get("pulls", 0))
            arm.reward_sum = float(item.get("reward_sum", 0.0))
            arm.ts_alpha = float(item.get("ts_alpha", 1.0))
            arm.ts_beta = float(item.get("ts_beta", 1.0))
            arm.recent_rewards = [float(x) for x in item.get("recent_rewards", [])]
        return b


@dataclass
class ConflictClause:
    problem_sig: str
    item_id: str  # cartridge_id, tactic_id, or patch_hash
    reason: str
    frequency: int = 1


class CdclConflictMemory:
    """
    Conflict-Driven Nogood Clause Memory (`CdclClauseDb` port for Python Apex search & agents).
    Prevents repeated exploration of cartridges, tactics, or code patches that already failed
    verification (SymPy contradiction / pytest failure / syntax error) on the same problem context.
    """

    def __init__(self, block_threshold: int = 1, enabled: bool = True):
        self.block_threshold = max(1, int(block_threshold))
        self.enabled = bool(enabled)
        self._clauses: Dict[Tuple[str, str], ConflictClause] = {}

    def record_conflict(self, problem_sig: str, item_id: str, reason: str = "verification_failed") -> ConflictClause:
        key = (str(problem_sig), str(item_id))
        if key in self._clauses:
            self._clauses[key].frequency += 1
            self._clauses[key].reason = str(reason)
        else:
            self._clauses[key] = ConflictClause(
                problem_sig=str(problem_sig),
                item_id=str(item_id),
                reason=str(reason),
                frequency=1,
            )
        return self._clauses[key]

    def is_blocked(self, problem_sig: str, item_id: str) -> bool:
        if not self.enabled:
            return False
        cl = self._clauses.get((str(problem_sig), str(item_id)))
        return bool(cl is not None and cl.frequency >= self.block_threshold)

    def blocked_items_for(self, problem_sig: str) -> Set[str]:
        if not self.enabled:
            return set()
        ps = str(problem_sig)
        return {
            cl.item_id
            for (sig, _), cl in self._clauses.items()
            if sig == ps and cl.frequency >= self.block_threshold
        }


@dataclass
class RolloutBranchPlan:
    branch_idx: int
    arm_label: str
    cartridge_weights: Dict[str, float]
    temperature: float
    use_ttt: bool = False


class CartridgeDiverseAllocator:
    """
    Successive Halving (`G=4 -> 8 -> 16`) branch allocator that decorrelates errors by assigning
    distinct orthogonal `.qmem v3` cartridge combinations across parallel/sequential rollouts.
    """

    def __init__(
        self,
        bandit: Optional[ContextualCartridgeBandit] = None,
        cdcl: Optional[CdclConflictMemory] = None,
        initial_g: int = 4,
        max_g: int = 16,
        agreement_stop_threshold: float = 0.60,
        enabled: bool = True,
    ):
        self.bandit = bandit
        self.cdcl = cdcl
        self.initial_g = max(2, int(initial_g))
        self.max_g = max(self.initial_g, int(max_g))
        self.agreement_stop_threshold = float(agreement_stop_threshold)
        self.enabled = bool(enabled)

    def plan_stage_branches(
        self,
        problem_sig: str,
        macro_domain: str,
        hgpr_weights: Dict[str, float],
        stage_size: Optional[int] = None,
        ttt_domain_id: Optional[str] = None,
        start_idx: int = 0,
    ) -> List[RolloutBranchPlan]:
        g = int(stage_size if stage_size is not None else self.initial_g)
        blocked = self.cdcl.blocked_items_for(problem_sig) if self.cdcl is not None else set()

        if self.bandit is not None:
            ranked = self.bandit.rank_candidates(hgpr_weights, macro_domain=macro_domain, blocked_domains=blocked)
        else:
            ranked = sorted(
                [(d, w) for d, w in hgpr_weights.items() if d not in blocked],
                key=lambda kv: (-kv[1], kv[0]),
            )

        top_doms = [d for d, _ in ranked]
        plans: List[RolloutBranchPlan] = []

        if not self.enabled or not top_doms:
            # Uniform base fallback (or TTT if available)
            for i in range(g):
                temp = 0.0 if (start_idx + i) == 0 else 0.6
                use_ttt = bool(ttt_domain_id and i == g - 1 and ttt_domain_id not in blocked)
                cw = {ttt_domain_id: 1.0} if use_ttt and ttt_domain_id else {}
                plans.append(RolloutBranchPlan(start_idx + i, "ttt" if use_ttt else "base", cw, temp, use_ttt))
            return plans

        # Build diverse portfolio of orthogonal cartridge configs to break correlated failure modes
        portfolio: List[Tuple[str, Dict[str, float], float, bool]] = []
        d0 = top_doms[0]
        portfolio.append((f"top1:{d0}", {d0: 1.0}, 0.0, False))
        if len(top_doms) >= 2:
            d1 = top_doms[1]
            portfolio.append((f"blend:{d0}+{d1}", {d0: 0.65, d1: 0.35}, 0.5, False))
            portfolio.append((f"top2:{d1}", {d1: 1.0}, 0.6, False))
        else:
            portfolio.append((f"top1_stoch:{d0}", {d0: 1.0}, 0.6, False))

        if ttt_domain_id and ttt_domain_id not in blocked:
            portfolio.append((f"ttt:{ttt_domain_id}", {ttt_domain_id: 1.0}, 0.45, True))
            portfolio.append((f"blend_ttt:{d0}+{ttt_domain_id}", {d0: 0.5, ttt_domain_id: 0.5}, 0.55, True))
        else:
            portfolio.append(("base_control", {}, 0.6, False))

        if len(top_doms) >= 3:
            d2 = top_doms[2]
            portfolio.append((f"tri_blend:{d0}+{top_doms[1]}+{d2}", {d0: 0.5, top_doms[1]: 0.3, d2: 0.2}, 0.6, False))

        for i in range(g):
            label, cw, temp, use_ttt = portfolio[(start_idx + i) % len(portfolio)]
            # Vary temperature slightly on later escalation rounds so rollouts stay diverse
            if start_idx + i >= len(portfolio) and temp == 0.0:
                temp = 0.65
            plans.append(RolloutBranchPlan(start_idx + i, label, dict(cw), temp, use_ttt))
        return plans

    def should_escalate(
        self,
        current_total_g: int,
        max_agreement_ratio: float,
        has_symbolic_contradiction: bool,
    ) -> Tuple[bool, int]:
        """
        Successive Halving gate: escalates G=4 -> 8 -> 16 ONLY when consensus is weak
        or verifiable contradictions remain, keeping average latency near G=4.
        """
        if current_total_g >= self.max_g:
            return False, 0
        if (max_agreement_ratio < self.agreement_stop_threshold) or has_symbolic_contradiction:
            next_chunk = min(current_total_g, self.max_g - current_total_g)
            return True, max(1, next_chunk)
        return False, 0
