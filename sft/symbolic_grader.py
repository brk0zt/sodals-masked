"""
SODALS v2.1 — Module 3.1: Symbolic Exact Match Grader
=====================================================
Replaces fragile substring matching with a 4-tier symbolic grading system.

Tiers:
  1. Normalized string match (fast, exact)
  2. SymPy symbolic equivalence (algebraic simplification)
  3. Numeric evaluation (parametric expressions)
  4. Abelian group isomorphism (group theory objects)

Usage:
  grader = SymbolicExactMatchGrader()
  result = grader.grade("\\\\mathbb{Z}^5", "Z+Z+Z+Z+Z")
  # GradeResult(match=True, method='group_iso', confidence=0.90, ...)
"""

import dataclasses
import re
import sys
from typing import List, Tuple, Optional, Dict

# Graceful SymPy import
try:
    from sympy import sympify, simplify, Symbol, symbols, Rational, oo
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


@dataclasses.dataclass
class GradeResult:
    match: bool
    method: str       # 'exact_string', 'symbolic_equiv', 'numeric_eval', 'group_iso', 'no_match'
    confidence: float  # 1.0, 0.95, 0.85, 0.90, 0.0
    details: str = ''


class SymbolicExactMatchGrader:
    """4-tier symbolic mathematics grader for benchmark evaluation."""

    # LaTeX command stripping patterns
    _LATEX_STRIP = [
        (r'\\mathbb\{([A-Z])\}', r'\1'),      # \mathbb{Z} -> Z
        (r'\\mathfrak\{([a-z])\}', r'\1'),     # \mathfrak{g} -> g
        (r'\\mathrm\{([^}]+)\}', r'\1'),       # \mathrm{Ext} -> Ext
        (r'\\text\{([^}]+)\}', r'\1'),         # \text{...} -> ...
        (r'\\mathcal\{([A-Z])\}', r'\1'),      # \mathcal{A} -> A
        (r'\\tilde\{([^}]+)\}', r'\1'),        # \tilde{X} -> X
        (r'\\hat\{([^}]+)\}', r'\1'),          # \hat{X} -> X
        (r'\\overline\{([^}]+)\}', r'\1'),     # \overline{X} -> X
        (r'\\left', ''), (r'\\right', ''),
        (r'\\[,;:!]', ''),                     # spacing commands
        (r'\\cdot', '*'), (r'\\times', '*'),
        (r'\\oplus', '+'), (r'\\bigoplus', '+'),
        (r'\\cong', '='), (r'\\simeq', '='), (r'\\approx', '='),
        (r'\\quad', ' '), (r'\\qquad', ' '),
        (r'\\pi', 'pi'), (r'\\infty', 'oo'),
        (r'\$', ''),
    ]

    # Abelian group parsing patterns
    _GROUP_PATTERNS = {
        'free_power':   re.compile(r'[Zℤ]\s*\^\s*(\d+)'),                            # Z^5
        'free_sum':     re.compile(r'(?:[Zℤ]\s*\+\s*){2,}[Zℤ]'),                     # Z+Z+Z
        'cyclic':       re.compile(r'[Zℤ]\s*/\s*(\d+)\s*[Zℤ]'),                      # Z/12Z
        'cyclic_sub':   re.compile(r'[Zℤ]\s*_\s*(\d+)'),                              # Z_2
        'cyclic_mod':   re.compile(r'[Zℤ]\s*/\s*(\d+)'),                              # Z/12
    }

    def grade(self, model_answer: str, ground_truth: str) -> GradeResult:
        """Grade a model answer against ground truth using 4-tier evaluation."""
        if not model_answer or not ground_truth:
            return GradeResult(match=False, method='no_match', confidence=0.0,
                               details='Empty answer or ground truth')

        # Tier 1: Normalized string match
        result = self._tier1_string_match(model_answer, ground_truth)
        if result.match:
            return result

        # Tier 4: Abelian group isomorphism (before sympy — fast and common)
        result = self._tier4_group_isomorphism(model_answer, ground_truth)
        if result.match:
            return result

        if not SYMPY_AVAILABLE:
            return GradeResult(match=False, method='no_match', confidence=0.0,
                               details='SymPy not available; Tier 1 & 4 failed')

        # Tier 2: SymPy symbolic equivalence
        result = self._tier2_symbolic_equiv(model_answer, ground_truth)
        if result.match:
            return result

        # Tier 3: Numeric evaluation
        result = self._tier3_numeric_eval(model_answer, ground_truth)
        if result.match:
            return result

        return GradeResult(match=False, method='no_match', confidence=0.0,
                           details='All 4 tiers failed')

    def grade_batch(self, pairs: List[Tuple[str, str]]) -> List[GradeResult]:
        """Grade multiple (model_answer, ground_truth) pairs."""
        return [self.grade(m, g) for m, g in pairs]

    # ── Tier 1: Normalized String Match ──────────────────────────────

    def _normalize(self, text: str) -> str:
        """Strip LaTeX commands, whitespace, and normalize."""
        s = text.strip()
        for pattern, replacement in self._LATEX_STRIP:
            s = re.sub(pattern, replacement, s)
        s = re.sub(r'\s+', '', s)  # collapse whitespace
        s = s.lower().strip()
        return s

    @staticmethod
    def _strip_outer_parens(s: str) -> str:
        s = s.strip()
        while s.startswith("(") and s.endswith(")"):
            depth = 0
            closed_early = False
            for c in s[:-1]:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        closed_early = True
                        break
            if not closed_early and depth == 1:
                s = s[1:-1].strip()
            else:
                break
        return s

    def _tier1_string_match(self, model: str, gt: str) -> GradeResult:
        nm, ng = self._normalize(model), self._normalize(gt)
        if nm == ng:
            return GradeResult(match=True, method='exact_string',
                               confidence=1.0, details=f'Normalized: "{nm}"')
        # Only try stripping redundant outermost matching parentheses (e.g. "(42)" vs "42")
        nm2 = self._strip_outer_parens(nm)
        ng2 = self._strip_outer_parens(ng)
        if nm2 == ng2:
            return GradeResult(match=True, method='exact_string',
                               confidence=0.98, details='Match after outer paren removal')
        return GradeResult(match=False, method='exact_string', confidence=0.0)

    # ── Tier 2: SymPy Symbolic Equivalence ───────────────────────────

    def _convert_fractions_and_roots(self, text: str) -> str:
        s = text
        # Convert nested \frac{a}{b} -> ((a)/(b))
        while r"\frac" in s or r"\dfrac" in s:
            s = s.replace(r"\dfrac", r"\frac")
            idx = s.find(r"\frac")
            open1 = s.find("{", idx)
            if open1 == -1 or open1 > idx + 8:
                m = re.match(r'\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])', s[idx:])
                if m:
                    s = s[:idx] + f"(({m.group(1)})/({m.group(2)}))" + s[idx + len(m.group(0)):]
                    continue
                break
            depth = 0
            close1 = -1
            for i in range(open1, len(s)):
                if s[i] == "{": depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        close1 = i
                        break
            if close1 == -1: break
            open2 = s.find("{", close1 + 1)
            if open2 == -1 or open2 > close1 + 3: break
            depth = 0
            close2 = -1
            for i in range(open2, len(s)):
                if s[i] == "{": depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        close2 = i
                        break
            if close2 == -1: break
            num = s[open1 + 1:close1]
            den = s[open2 + 1:close2]
            s = s[:idx] + f"(({num})/({den}))" + s[close2 + 1:]

        # Convert \sqrt{x} -> sqrt(x)
        while r"\sqrt" in s:
            idx = s.find(r"\sqrt")
            open_b = s.find("{", idx)
            if open_b == -1 or open_b > idx + 8:
                m = re.match(r'\\sqrt\s*([0-9a-zA-Z])', s[idx:])
                if m:
                    s = s[:idx] + f"sqrt({m.group(1)})" + s[idx + len(m.group(0)):]
                    continue
                break
            depth = 0
            close_b = -1
            for i in range(open_b, len(s)):
                if s[i] == "{": depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        close_b = i
                        break
            if close_b == -1: break
            inside = s[open_b + 1:close_b]
            s = s[:idx] + f"sqrt({inside})" + s[close_b + 1:]
        return s

    def _latex_to_sympy_str(self, text: str) -> str:
        """Convert LaTeX-ish math to SymPy-parseable string."""
        s = text.strip()
        # Strip Answer: prefix
        s = re.sub(r'^Answer:\s*', '', s, flags=re.IGNORECASE)
        # Convert LaTeX fractions and roots before stripping backslashes
        s = self._convert_fractions_and_roots(s)
        # LaTeX to algebra
        for pattern, replacement in self._LATEX_STRIP:
            s = re.sub(pattern, replacement, s)
        # Power notation: x^2 -> x**2
        s = re.sub(r'\^(\d+)', r'**\1', s)
        s = re.sub(r'\^\{([^}]+)\}', r'**(\1)', s)
        # Implicit multiplication: 4x -> 4*x, but not 'pi'
        s = re.sub(r'(\d)([a-wyzA-Z])', r'\1*\2', s)
        # Remove remaining backslashes
        s = re.sub(r'\\', '', s)
        s = s.strip()
        return s

    def _safe_parse(self, text: str):
        """Safely parse an expression string to SymPy."""
        try:
            s = self._latex_to_sympy_str(text)
            if not s:
                return None
            return parse_expr(s, transformations=standard_transformations + (implicit_multiplication_application,))
        except Exception:
            try:
                return sympify(s)
            except Exception:
                return None

    def _tier2_symbolic_equiv(self, model: str, gt: str) -> GradeResult:
        try:
            expr_m = self._safe_parse(model)
            expr_g = self._safe_parse(gt)
            if expr_m is None or expr_g is None:
                return GradeResult(match=False, method='symbolic_equiv',
                                   confidence=0.0, details='Parse failed')
            diff = simplify(expr_m - expr_g)
            if diff == 0:
                return GradeResult(match=True, method='symbolic_equiv',
                                   confidence=0.95,
                                   details=f'simplify({expr_m} - {expr_g}) = 0')
            # Try expanding
            from sympy import expand
            if expand(expr_m - expr_g) == 0:
                return GradeResult(match=True, method='symbolic_equiv',
                                   confidence=0.93,
                                   details=f'expand({expr_m} - {expr_g}) = 0')
        except Exception as e:
            return GradeResult(match=False, method='symbolic_equiv',
                               confidence=0.0, details=f'Error: {e}')
        return GradeResult(match=False, method='symbolic_equiv', confidence=0.0)

    # ── Tier 3: Numeric Evaluation ───────────────────────────────────

    def _tier3_numeric_eval(self, model: str, gt: str) -> GradeResult:
        try:
            expr_m = self._safe_parse(model)
            expr_g = self._safe_parse(gt)
            if expr_m is None or expr_g is None:
                return GradeResult(match=False, method='numeric_eval',
                                   confidence=0.0, details='Parse failed')

            # Find free symbols
            free_vars = expr_m.free_symbols | expr_g.free_symbols
            if not free_vars:
                # Both are constants — just evaluate
                try:
                    vm = complex(expr_m.evalf())
                    vg = complex(expr_g.evalf())
                    if abs(vm - vg) < 1e-6:
                        return GradeResult(match=True, method='numeric_eval',
                                           confidence=0.90,
                                           details=f'{vm} ≈ {vg}')
                except Exception:
                    pass
                return GradeResult(match=False, method='numeric_eval',
                                   confidence=0.0)

            # Generate test points
            import random
            random.seed(42)
            test_values = [random.uniform(1.0, 10.0) for _ in range(10)]
            var_list = sorted(free_vars, key=str)

            all_close = True
            for tv in test_values:
                subs = {v: tv + i * 0.7 for i, v in enumerate(var_list)}
                try:
                    vm = complex(expr_m.subs(subs).evalf())
                    vg = complex(expr_g.subs(subs).evalf())
                    if abs(vm - vg) > 1e-4:
                        all_close = False
                        break
                except Exception:
                    all_close = False
                    break

            if all_close:
                return GradeResult(match=True, method='numeric_eval',
                                   confidence=0.85,
                                   details=f'10 test points match for vars {[str(v) for v in var_list]}')
        except Exception as e:
            return GradeResult(match=False, method='numeric_eval',
                               confidence=0.0, details=f'Error: {e}')
        return GradeResult(match=False, method='numeric_eval', confidence=0.0)

    # ── Tier 4: Abelian Group Isomorphism ────────────────────────────

    @staticmethod
    def _elementary_divisors(factors: List[int]) -> Tuple[int, Tuple[int, ...]]:
        """
        Computes the elementary divisors (primary decomposition) of a finitely generated abelian group.
        0 = Z (free rank)
        1 = Z/1Z = {0} (trivial group, omitted)
        n > 1 is factored into prime powers p^k.
        Returns: (free_rank, sorted_tuple_of_prime_powers)
        """
        free_rank = factors.count(0)
        prime_powers = []
        for n in factors:
            if n <= 1:
                continue
            temp = n
            d = 2
            while d * d <= temp:
                if temp % d == 0:
                    pk = 1
                    while temp % d == 0:
                        pk *= d
                        temp //= d
                    prime_powers.append(pk)
                d += 1
            if temp > 1:
                prime_powers.append(temp)
        return free_rank, tuple(sorted(prime_powers))

    def _parse_abelian_group(self, text: str) -> Optional[List[int]]:
        """
        Parse an abelian group expression into direct sum factors:
        0 = Z (infinite cyclic), n > 0 = Z/nZ (cyclic torsion).
        
        Examples:
          Z^5            -> [0,0,0,0,0]
          Z+Z+Z+Z+Z     -> [0,0,0,0,0]
          Z/12Z          -> [12]
          Z^2 ⊕ Z/3Z    -> [0,0,3]
          Z_2            -> [2]
          Z              -> [0]
        """
        s = text.strip()
        # Normalize group direct sum operators before stripping LaTeX
        s = s.replace(r"\oplus", "+").replace(r"\bigoplus", "+").replace(r"\times", "+")
        s = s.replace("⊕", "+").replace("×", "+").replace("*", "+")

        # Strip LaTeX formatting
        for pattern, replacement in self._LATEX_STRIP:
            s = re.sub(pattern, replacement, s)
        s = s.strip()

        # Check if it looks like a group expression
        if not re.search(r'[Zℤ]', s):
            return None

        factors = []
        # Split strictly by '+' (direct sum)
        parts = [p.strip() for p in s.split("+")]
        if len(parts) == 1 and not re.search(r'[\^/_]', s):
            # Single Z
            if re.match(r'^[Zℤ]$', s.strip()):
                return [0]
            return None

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Z^n
            m = re.match(r'^[Zℤ]\s*\^\s*\{?(\d+)\}?$', part)
            if m:
                n = int(m.group(1))
                factors.extend([0] * n)
                continue

            # Z/nZ or Z/n
            m = re.match(r'^[Zℤ]\s*/\s*(\d+)\s*[Zℤ]?$', part)
            if m:
                factors.append(int(m.group(1)))
                continue

            # Z_n (subscript = torsion)
            m = re.match(r'^[Zℤ]\s*_\s*\{?(\d+)\}?$', part)
            if m:
                factors.append(int(m.group(1)))
                continue

            # Plain Z
            if re.match(r'^[Zℤ]$', part):
                factors.append(0)
                continue

            # Unrecognized part
            return None

        return factors if factors else None

    def _groups_isomorphic(self, g1: List[int], g2: List[int]) -> bool:
        """
        Two finitely-generated abelian groups are isomorphic iff their primary decompositions
        (free rank and elementary prime-power divisors) are equal (CRT / Fundamental Theorem).
        """
        return self._elementary_divisors(g1) == self._elementary_divisors(g2)

    def _tier4_group_isomorphism(self, model: str, gt: str) -> GradeResult:
        g1 = self._parse_abelian_group(model)
        g2 = self._parse_abelian_group(gt)
        if g1 is None or g2 is None:
            return GradeResult(match=False, method='group_iso',
                               confidence=0.0, details='Not parseable as abelian groups')
        if self._groups_isomorphic(g1, g2):
            return GradeResult(match=True, method='group_iso',
                               confidence=0.90,
                               details=f'Groups isomorphic: {g1} ≅ {g2}')
        return GradeResult(match=False, method='group_iso', confidence=0.0,
                           details=f'Not isomorphic: {g1} ≇ {g2}')


# ═══════════════════════════════════════════════════════════════════
#  Self-Test Suite
# ═══════════════════════════════════════════════════════════════════

def run_self_test():
    sys.stdout.reconfigure(encoding='utf-8')
    grader = SymbolicExactMatchGrader()
    
    test_cases = [
        # (model_answer, ground_truth, expected_match, description)
        ('18', '18', True, 'Exact numeric match'),
        ('3', '3', True, 'Simple integer match'),
        ('\\mathbb{Z}', 'Z', True, 'LaTeX Z vs plain Z — group iso'),
        ('Z+Z+Z+Z+Z', '\\mathbb{Z}^5', True, 'Z sum vs Z^5 — group iso'),
        ('\\mathbb{Z}^5', 'Z+Z+Z+Z+Z', True, 'Z^5 vs Z sum — group iso (reverse)'),
        ('\\mathbb{Z}/12\\mathbb{Z}', '\\mathbb{Z}', False,
         'CRITICAL: Z/12Z ≠ Z — old grader false positive'),
        ('\\mathbb{Z}_2', '\\mathbb{Z}', False, 'Z_2 ≠ Z — torsion vs free'),
        ('1+6x+4x^2+4x^3+4x^4+6x^5+x^6',
         '1+3x+6x^2+8x^3+6x^4+3x^5+x^6', False,
         'Different polynomials — must NOT match'),
        ('(-1)^k (d-2k)', 'd-(d-2k)^2', False,
         'Different expressions in d and k'),
        ('-4\\pi', '-4\\pi', True, 'Exact LaTeX match'),
        ('\\mathbb{Z}/3\\mathbb{Z}', 'Z/3Z', True, 'Z/3Z — group iso notation variants'),
    ]

    # SymPy-dependent tests
    if SYMPY_AVAILABLE:
        test_cases.extend([
            ('d-(d-2*k)^2', '-(d-2*k)**2 + d', True,
             'Symbolic equivalence: reordered terms'),
            ('2', '2', True, 'Trivial symbolic match'),
        ])

    passed = 0
    failed = 0
    print("=" * 72)
    print("  SODALS v2.1 — Symbolic Grader Self-Test Suite")
    print("=" * 72)
    print(f"  SymPy available: {SYMPY_AVAILABLE}")
    print()

    for model_ans, gt, expected, desc in test_cases:
        result = grader.grade(model_ans, gt)
        ok = result.match == expected
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"  {status} | {desc}")
        print(f"         Model: {model_ans!r}")
        print(f"         GT:    {gt!r}")
        print(f"         Got:   match={result.match}, method={result.method}, "
              f"conf={result.confidence:.2f}")
        if result.details:
            print(f"         Info:  {result.details}")
        print()

    print("=" * 72)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  🎉 ALL TESTS PASSED")
    else:
        print("  ⚠️  SOME TESTS FAILED")
    print("=" * 72)
    return failed == 0


# Alias for backward compatibility
SymbolicGrader = SymbolicExactMatchGrader


if __name__ == '__main__':
    sys.exit(0 if run_self_test() else 1)
