"""
Production-Ready Unit Test Suite: Module 3 (CST Symbolic Verifier & 4-Tier Symbolic Grader)
=============================================================================================
Validates:
  1. Strict defense against False Positives (e.g. \frac{1}{2} != 2, \sqrt{2} != 2, (1+2)*3 != 1+(2*3)).
  2. Safe unit stripping (no corruption of variable 'm' or LaTeX '\binom' -> '\bino').
  3. SymPy LaTeX fraction & root parsing in Tier 2 / Tier 3.
  4. Exact CRT & Elementary Divisor Abelian Group Isomorphism in Tier 4.
"""

import os
import sys
import unittest

SFT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.exists(os.path.join(SFT_DIR, "cst_symbolic_verifier.py")):
    SFT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
    SFT_DIR = os.path.abspath(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from cst_symbolic_verifier import NormalizedMathMatcher
from symbolic_grader import SymbolicGrader, GradeResult


class TestModule3SymbolicVerifiers(unittest.TestCase):

    def setUp(self):
        self.grader = SymbolicGrader()

    def test_false_positive_rejection_in_cst_matcher(self):
        """Validates that wrong expressions sharing trailing digits are NOT graded as correct."""
        gt_int = "2"
        wrong_preds = [
            r"\frac{1}{2}",
            r"\sqrt{2}",
            "x^2",
            "3^2",
            "10+2",
            "-2",
            r"\frac{3}{2}",
            r"\log(2)",
        ]
        for pred in wrong_preds:
            strict, sym = NormalizedMathMatcher.check_match(pred, gt_int)
            self.assertFalse(
                sym,
                f"Catastrophic False Positive: pred '{pred}' was accepted as matching gt '{gt_int}'!"
            )

    def test_unit_stripping_and_latex_integrity(self):
        """Validates that physical units are stripped without corrupting variable 'm' or '\binom'."""
        # 1. Physical units should be cleanly stripped
        self.assertEqual(NormalizedMathMatcher.clean_latex("5 cm"), "5")
        self.assertEqual(NormalizedMathMatcher.clean_latex("12 inches"), "12")
        self.assertEqual(NormalizedMathMatcher.clean_latex("10 m"), "10")
        self.assertEqual(NormalizedMathMatcher.clean_latex("45 degrees"), "45")

        # 2. Variable 'm' and algebraic symbols must be preserved!
        self.assertEqual(NormalizedMathMatcher.clean_latex("m"), "m")
        self.assertEqual(NormalizedMathMatcher.clean_latex("x + m"), "x+m")
        self.assertEqual(NormalizedMathMatcher.clean_latex("2m"), "2m")
        self.assertEqual(NormalizedMathMatcher.clean_latex(r"\binom{n}{k}"), r"\binom{n}{k}")

    def test_true_equivalences_pass_cst_matcher(self):
        """Validates that mathematically equal expressions correctly pass."""
        equiv_pairs = [
            ("2", "2"),
            ("2.0", "2"),
            ("x = 2", "2"),
            (r"\frac{2}{4}", r"\frac{1}{2}"),
            (r"(x+1)^2", r"x^2 + 2x + 1"),
            (r"\frac{x^2 - 1}{x - 1}", "x + 1"),
            (r"\frac{\sqrt{2}}{2}", r"\frac{1}{\sqrt{2}}"),
        ]
        for pred, gt in equiv_pairs:
            strict, sym = NormalizedMathMatcher.check_match(pred, gt)
            self.assertTrue(
                sym,
                f"Valid equivalent pair failed match: pred '{pred}' vs gt '{gt}'"
            )

    def test_grader_tier1_parenthesis_operator_precedence(self):
        """Validates that Tier 1 does NOT equate expressions with differing operator precedence."""
        # (1 + 2) * 3 = 9 != 1 + (2 * 3) = 7
        res = self.grader.grade("(1 + 2) * 3", "1 + (2 * 3)")
        self.assertFalse(res.match, "(1+2)*3 must not match 1+(2*3)!")

        # Outer parentheses stripping should work
        res_outer = self.grader.grade("(42)", "42")
        self.assertTrue(res_outer.match, "(42) should match 42")

    def test_grader_tier2_latex_fraction_and_root(self):
        """Validates SymPy parsing of LaTeX fractions and roots."""
        res_frac = self.grader.grade(r"\frac{x^2 - 4}{x - 2}", "x + 2")
        self.assertTrue(res_frac.match)
        self.assertIn(res_frac.method, ["sympy_exact", "symbolic_equiv"])

        res_sqrt = self.grader.grade(r"\sqrt{12}", r"2\sqrt{3}")
        self.assertTrue(res_sqrt.match)

    def test_grader_tier4_abelian_group_isomorphism(self):
        """Validates CRT and Primary Decomposition / Elementary Divisors for Abelian Groups."""
        # Z/6Z ≅ Z/2Z ⊕ Z/3Z (CRT match)
        res_crt = self.grader.grade(r"\mathbb{Z}/6\mathbb{Z}", r"\mathbb{Z}/2\mathbb{Z} \oplus \mathbb{Z}/3\mathbb{Z}")
        self.assertTrue(res_crt.match, "Z/6Z must be isomorphic to Z/2Z ⊕ Z/3Z!")
        self.assertEqual(res_crt.method, "group_iso")

        # Z_12 ≅ Z_3 ⊕ Z_4
        res_12 = self.grader.grade(r"Z_{12}", r"Z_3 \oplus Z_4")
        self.assertTrue(res_12.match)

        # Z_12 ≇ Z_2 ⊕ Z_6 (non-isomorphic direct sums)
        res_non = self.grader.grade(r"Z_{12}", r"Z_2 \oplus Z_6")
        self.assertFalse(res_non.match, "Z_12 must NOT be isomorphic to Z_2 ⊕ Z_6!")

        # Trivial group elimination: Z_1 ⊕ Z^2 ≅ Z^2
        res_triv = self.grader.grade(r"Z_1 \oplus Z^2", r"Z^2")
        self.assertTrue(res_triv.match, "Z_1 ⊕ Z^2 must be isomorphic to Z^2!")

        # Free power: Z^3 ≅ Z + Z + Z
        res_free = self.grader.grade(r"Z^3", r"Z \oplus Z \oplus Z")
        self.assertTrue(res_free.match)


if __name__ == "__main__":
    unittest.main(verbosity=2)
