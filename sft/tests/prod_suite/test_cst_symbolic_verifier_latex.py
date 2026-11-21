"""
Production Unit Test Suite — CST Symbolic Verifier & NormalizedMathMatcher LaTeX Cleanup
========================================================================================
Tests:
  1. LaTeX spacing/thinning command removal (`\\,`, `\\!`, `\\;`, `\\:`, `\\quad`, `\\qquad`, `~`)
     ordered BEFORE thousands-separator comma stripping.
  2. Thousands-separator equivalence (`10,\\!080`, `10,080`, `1,234,567`) across clean_latex,
     float, and SymPy stages in `NormalizedMathMatcher.check_match`.
  3. Preservation of ordered pairs and intervals (`(3, 4)`, `(3, 400)`, `(-5, 100)`, `[1, 500]`)
     so tuples are never collapsed into concatenated integers.
  4. Preservation of non-thousands decimals (`0,125`, `3,14`) without breaking numeric values.
  5. `CstSymbolicVerifier.clean_latex_expr` and `verify_chunk` on `10,\\!080` and tuple checks.
"""

import os
import sys
import unittest

SFT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from cst_symbolic_verifier import CstSymbolicVerifier, LeanTacticStepVerifier, NormalizedMathMatcher, safe_sympify


class TestCstSymbolicVerifierLatex(unittest.TestCase):
    def test_q199_latex_thin_space_and_thousands_match(self):
        legacy, sym = NormalizedMathMatcher.check_match("10080", r"10,\!080")
        self.assertTrue(legacy)
        self.assertTrue(sym)

        legacy_rev, sym_rev = NormalizedMathMatcher.check_match(r"10,\!080", "10080")
        self.assertTrue(legacy_rev)
        self.assertTrue(sym_rev)

        legacy_plain, sym_plain = NormalizedMathMatcher.check_match("10080", "10,080")
        self.assertTrue(legacy_plain)
        self.assertTrue(sym_plain)

    def test_float_and_sympy_stages_with_thousands_and_spacing(self):
        # Float equivalence stage
        _, sym_float = NormalizedMathMatcher.check_match("10080.0", r"10,\!080")
        self.assertTrue(sym_float)

        # SymPy algebraic equivalence stage (must parse cleaned 10080, not tuple (10, 080))
        _, sym_expr = NormalizedMathMatcher.check_match("2 * 5040", r"10,\!080")
        self.assertTrue(sym_expr)

        # LaTeX expression containing formatted thousands inside fraction
        _, sym_frac = NormalizedMathMatcher.check_match("5040", r"\frac{10,\!080}{2}")
        self.assertTrue(sym_frac)

    def test_all_latex_spacing_commands_and_multigroup_thousands(self):
        variants = [
            r"10\,080",
            r"10\!080",
            r"10\;080",
            r"10\:080",
            r"10\quad080",
            r"10\qquad080",
            r"10~080",
            r"10,\!080",
            r"10,\,080",
            r"10,\quad080",
        ]
        for v in variants:
            with self.subTest(variant=v):
                self.assertEqual(NormalizedMathMatcher.clean_latex(v), "10080")
                _, sym = NormalizedMathMatcher.check_match("10080", v)
                self.assertTrue(sym)

        # Multi-group thousands separator (must not leave leftover commas like 1234,567)
        self.assertEqual(NormalizedMathMatcher.clean_latex(r"1,\!234,\!567"), "1234567")
        _, sym_big = NormalizedMathMatcher.check_match("1234567", r"1,\!234,\!567")
        self.assertTrue(sym_big)

    def test_ordered_pairs_and_intervals_preserved(self):
        pair_cases = [
            ("(3, 4)", "(3,4)"),
            ("(3, 400)", "(3,400)"),
            ("(-5, 100)", "(-5,100)"),
            ("[1, 500]", "[1,500]"),
            (r"\left(3,\; 400\right)", "(3,400)"),
        ]
        for a, b in pair_cases:
            with self.subTest(a=a, b=b):
                legacy, sym = NormalizedMathMatcher.check_match(a, b)
                self.assertTrue(legacy)
                self.assertTrue(sym)

        # Must NOT collapse (3, 400) into 3400 or (3400)
        self.assertEqual(NormalizedMathMatcher.clean_latex("(3, 400)"), "(3,400)")
        _, false_collapse = NormalizedMathMatcher.check_match("3400", "(3, 400)")
        self.assertFalse(false_collapse)

    def test_decimal_commas_not_corrupted_into_integers(self):
        # Leading zero decimal comma (0,125) and 2-digit fractional part (3,14)
        self.assertEqual(NormalizedMathMatcher.clean_latex("0,125"), "0,125")
        self.assertEqual(NormalizedMathMatcher.clean_latex("3,14"), "3,14")

        _, match_zero_dec = NormalizedMathMatcher.check_match("0,125", "125")
        self.assertFalse(match_zero_dec)
        _, match_pi_dec = NormalizedMathMatcher.check_match("3,14", "314")
        self.assertFalse(match_pi_dec)

    def test_cst_symbolic_verifier_clean_and_verify_chunk(self):
        self.assertEqual(CstSymbolicVerifier.clean_latex_expr(r"10,\!080"), "10080")
        self.assertEqual(CstSymbolicVerifier.clean_latex_expr(r"1,\!234,\!567"), "1234567")
        self.assertEqual(CstSymbolicVerifier.clean_latex_expr("(3, 400)"), "(3, 400)")

        ok, count, errors = CstSymbolicVerifier.verify_chunk(r"2 * 5040 = 10,\!080")
        self.assertTrue(ok)
        self.assertGreaterEqual(count, 1)
        self.assertEqual(errors, [])

        bad_ok, _, bad_errors = CstSymbolicVerifier.verify_chunk(r"10,\!080 = 10081")
        self.assertFalse(bad_ok)
        self.assertGreaterEqual(len(bad_errors), 1)

    def test_relative_tolerance_and_safe_sympify_security(self):
        # Irrational decimal rounding sqrt(2) = 1.414 should NOT raise false contradiction
        ok_sqrt, _, errs_sqrt = CstSymbolicVerifier.verify_chunk("sqrt(2) = 1.414")
        self.assertTrue(ok_sqrt)
        self.assertEqual(errs_sqrt, [])

        # Gross approximation pi = 3.1 MUST still be caught
        bad_pi, _, errs_pi = CstSymbolicVerifier.verify_chunk("pi = 3.1")
        self.assertFalse(bad_pi)
        self.assertGreaterEqual(len(errs_pi), 1)

        # Untrusted code/eval gadgets and exponent DoS must return None safely
        self.assertIsNone(safe_sympify("__import__('os').system('echo pwn')"))
        self.assertIsNone(safe_sympify("eval('1+1')"))
        self.assertIsNone(safe_sympify("(lambda x: x)(5)"))
        self.assertIsNone(safe_sympify("2**999999"))
        self.assertIsNotNone(safe_sympify("sqrt(2) + 3/4"))

    def test_mixed_brace_fractions_and_q167_degrees_unit(self):
        # Q167 regression: 270/7 vs \frac{270}7\text{ degrees}
        _, sym_q167 = NormalizedMathMatcher.check_match("270/7", r"\frac{270}7\text{ degrees}")
        self.assertTrue(sym_q167)

        _, sym_unbraced = NormalizedMathMatcher.check_match("0.5", r"\frac1{2}")
        self.assertTrue(sym_unbraced)

        _, sym_pair = NormalizedMathMatcher.check_match(r"\frac{1}2", r"\frac12")
        self.assertTrue(sym_pair)

    def test_degree_trigonometry_and_prompt_consistency_q165(self):
        # Incorrect sign on Q165 equation MUST be caught numerically by CST
        bad_ok, _, bad_errs = CstSymbolicVerifier.verify_chunk(
            r"\tan 100^\circ + 4 \sin 100^\circ = \sqrt{3}"
        )
        self.assertFalse(bad_ok)
        self.assertGreaterEqual(len(bad_errs), 1)

        # Correct negative identity MUST verify cleanly
        good_ok, good_cnt, good_errs = CstSymbolicVerifier.verify_chunk(
            r"\tan 100^\circ + 4 \sin 100^\circ = -\sqrt{3}"
        )
        self.assertTrue(good_ok)
        self.assertGreaterEqual(good_cnt, 1)
        self.assertEqual(good_errs, [])

        # Zero-leakage prompt target consistency check catches wrong-sign final boxed answer even when eq_count == 0
        q165_prompt = r"Simplify $\tan 100^\circ + 4 \sin 100^\circ.$"
        rep_wrong = LeanTacticStepVerifier.verify_proof_steps(
            r"<think>Using symmetry, sign cancels.</think>\n\boxed{\sqrt{3}}",
            prompt=q165_prompt,
        )
        self.assertFalse(rep_wrong.is_formally_sound)
        self.assertTrue(any("LeanGoalMismatchViolation" in c for c in rep_wrong.contradictions))

        rep_right = LeanTacticStepVerifier.verify_proof_steps(
            r"<think>Quadrant II: tan(100) < 0.</think>\n\boxed{-\sqrt{3}}",
            prompt=q165_prompt,
        )
        self.assertTrue(rep_right.is_formally_sound)
        self.assertEqual(rep_right.contradictions, ())

        # Non-trig pure degree arithmetic (Q069 regression: 180 - 134 = 46^\circ) must NOT raise false contradiction
        deg_ok, deg_cnt, deg_errs = CstSymbolicVerifier.verify_chunk(r"180 - 134 = 46^\circ")
        self.assertTrue(deg_ok)
        self.assertGreaterEqual(deg_cnt, 1)
        self.assertEqual(deg_errs, [])


if __name__ == "__main__":
    unittest.main()
