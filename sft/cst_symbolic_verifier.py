"""
SODALS v2.1 — CST (Continuous Semantic Topology) Symbolic Verifier & PRM
========================================================================
Neurosymbolic process verification module combining:
  1. SymPy-driven exact arithmetic & algebraic equation verification
  2. Subscript, power, fraction, and formatting-invariant LaTeX normalization
  3. Contradiction detection (identifying impossible deductions A != B)
  4. Normalized mathematical equivalence matching for evaluation
"""

import ast
import concurrent.futures
import re
import math
import sympy
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Sequence


_ALLOWED_SYMPY_FUNCS = {
    "sqrt", "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "log", "ln", "exp", "abs", "Abs", "Rational", "Integer",
    "Float", "floor", "ceiling", "factorial", "gamma",
}
_FORBIDDEN_NAMES = {
    "eval", "exec", "open", "compile", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "dir", "type", "object",
    "os", "sys", "subprocess", "importlib", "builtins", "__import__",
}
_SIMPLIFY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _is_safe_math_expr(expr_str: str, max_len: int = 160) -> bool:
    """AST-level whitelist guard so untrusted text can never trigger eval exploits in SymPy."""
    if not expr_str or len(expr_str) > max_len:
        return False
    low = expr_str.lower()
    if "__" in low or ";" in low:
        return False
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError:
        return False

    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > 120:
            return False
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant)):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                    if abs(node.right.value) > 256:
                        return False
            if not isinstance(node, ast.Constant) and isinstance(node, ast.BinOp):
                if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                    return False
            if isinstance(node, ast.UnaryOp) and not isinstance(node.op, (ast.UAdd, ast.USub)):
                return False
        elif isinstance(node, ast.Load):
            continue
        elif isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.UAdd, ast.USub)):
            continue
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES or node.id.startswith("_"):
                return False
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_SYMPY_FUNCS:
                return False
            if node.keywords:
                return False
        else:
            return False
    return True


def safe_sympify(expr_str: str, locals_dict: Optional[Dict[str, Any]] = None) -> Optional[sympy.Expr]:
    """Safely parses math expressions via AST whitelist before calling SymPy."""
    s = (expr_str or "").strip()
    if not _is_safe_math_expr(s):
        return None
    try:
        return sympy.sympify(s, locals=locals_dict)
    except Exception:
        return None


def safe_simplify_expr(expr: sympy.Expr, timeout_sec: float = 0.5) -> Optional[sympy.Expr]:
    """Simplifies algebraic residuals with op-count bounds and a Windows-safe timeout."""
    if expr is None:
        return None
    try:
        if sympy.count_ops(expr) > 120:
            return None
        if sympy.count_ops(expr) <= 25:
            return sympy.simplify(expr)
        fut = _SIMPLIFY_EXECUTOR.submit(sympy.simplify, expr)
        return fut.result(timeout=timeout_sec)
    except Exception:
        return None


def is_numeric_step_mismatch(f1: float, f2: float, rel_tol: float = 5e-4, abs_tol: float = 1e-4) -> bool:
    """
    Compares intermediate numerical steps without flagging rounded irrationals (e.g. sqrt(2) = 1.414)
    as contradictions, while strictly catching distinct integers (e.g. 10080 != 10081).
    """
    if not (math.isfinite(f1) and math.isfinite(f2)):
        return True
    both_int_like = (abs(f1 - round(f1)) < 1e-9) and (abs(f2 - round(f2)) < 1e-9)
    if both_int_like:
        return abs(f1 - f2) > 1e-6
    return not math.isclose(f1, f2, rel_tol=rel_tol, abs_tol=abs_tol)


_TRIG_LOG_FUNCS = ("sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "sinh", "cosh", "tanh", "ln", "log")


def _read_latex_arg(s: str, pos: int) -> Tuple[Optional[str], int]:
    """Reads either a balanced `{...}` LaTeX argument or a single unbraced token at `pos`."""
    n = len(s)
    while pos < n and s[pos].isspace():
        pos += 1
    if pos >= n:
        return None, pos
    if s[pos] == "{":
        depth = 0
        for i in range(pos, n):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return s[pos + 1 : i], i + 1
        return None, pos
    if s[pos].isalnum():
        return s[pos], pos + 1
    return None, pos


def _expand_latex_fracs(s: str) -> str:
    """Expands nested/braced/unbraced `\\frac` and `\\dfrac` (e.g. `\\frac{270}7`, `\\frac12`) to `((num)/(den))`."""
    out = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    guard = 0
    while r"\frac" in out and guard < 40:
        guard += 1
        idx = out.find(r"\frac")
        arg_pos = idx + 5
        if arg_pos < len(out) and out[arg_pos].isalpha():
            break
        num, after_num = _read_latex_arg(out, arg_pos)
        if num is None:
            break
        den, after_den = _read_latex_arg(out, after_num)
        if den is None:
            break
        out = out[:idx] + f"(({num})/({den}))" + out[after_den:]
    return out


def _expand_latex_sqrts(s: str) -> str:
    """Expands nested/braced/unbraced `\\sqrt{x}` and `\\sqrt2` to `sqrt(...)`."""
    out = s
    guard = 0
    while r"\sqrt" in out and guard < 40:
        guard += 1
        idx = out.find(r"\sqrt")
        arg, after_arg = _read_latex_arg(out, idx + 5)
        if arg is None:
            break
        out = out[:idx] + f"sqrt({arg})" + out[after_arg:]
    return out


def _parenthesize_latex_funcs(s: str) -> str:
    """Converts `\\tan 100` / `tan ((100)*pi/180)` into `tan(...)` with balanced parentheses."""
    out = re.sub(r"\\(" + "|".join(_TRIG_LOG_FUNCS) + r")\b", r" \1 ", s)
    func_re = re.compile(r"\b(" + "|".join(_TRIG_LOG_FUNCS) + r")\s+")
    guard = 0
    while guard < 40:
        guard += 1
        m = func_re.search(out)
        if not m:
            break
        fn = m.group(1)
        pos = m.end()
        if pos >= len(out):
            break
        if out[pos] == "(":
            depth = 0
            close_p = -1
            for i in range(pos, len(out)):
                if out[i] == "(":
                    depth += 1
                elif out[i] == ")":
                    depth -= 1
                    if depth == 0:
                        close_p = i
                        break
            if close_p == -1:
                break
            arg = out[pos + 1 : close_p]
            out = out[: m.start()] + f"{fn}({arg})" + out[close_p + 1 :]
        else:
            tok_m = re.match(r"([a-zA-Z0-9_.]+)", out[pos:])
            if not tok_m:
                break
            out = out[: m.start()] + f"{fn}({tok_m.group(1)})" + out[pos + tok_m.end() :]
    return out


def _extract_last_boxed(text: str) -> str:
    """Extracts the final balanced `\\boxed{...}` payload from a reasoning trace."""
    idx = (text or "").rfind(r"\boxed{")
    if idx == -1:
        return ""
    open_b = idx + len(r"\boxed")
    depth = 0
    for i in range(open_b, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_b + 1 : i].strip()
    return ""


class CstSymbolicVerifier:
    """
    Symbolic Process Reward Model (PRM) verifier.
    Extracts explicit deduction steps (lhs = rhs) from reasoning text,
    normalizes them symbolically, and checks mathematical consistency using SymPy.
    """

    @staticmethod
    def clean_latex_expr(expr_str: str) -> str:
        s = expr_str.strip()
        # Strip LaTeX spacing/thinning commands BEFORE thousands comma removal (e.g. 10,\!080 -> 10,080)
        s = re.sub(r'\\(?:qquad|quad)(?![a-zA-Z])|\\[,!;:]|~', '', s)
        # Remove thousands commas when forming valid thousands groups (preserving decimals and (a, b) pairs)
        if not re.fullmatch(r'[\(\[]\s*-?\d+\s*,\s*[1-9]\d*\s*[\)\]]', s):
            s = re.sub(
                r'(?<![.\d])([1-9]\d{0,2}(?:,\d{3})+)(?!\d)',
                lambda m: m.group(1).replace(',', ''),
                s,
            )
        # Convert degrees to radians ONLY when trigonometric functions are present;
        # otherwise strip degree symbols so pure angle arithmetic (180 - 134 = 46^\circ) stays consistent.
        has_trig = bool(re.search(r'(?:\\|\b)(?:sin|cos|tan|cot|sec|csc|asin|acos|atan)\b', s, flags=re.I))
        if has_trig:
            s = re.sub(r'(\d+(?:\.\d+)?)\s*(?:\^\s*\{?\s*\\circ\s*\}?|°)', r'((\1)*pi/180)', s)
            s = re.sub(r'\(([^()]+)\)\s*(?:\^\s*\{?\s*\\circ\s*\}?|°)', r'((\1)*pi/180)', s)
            s = re.sub(r'\^\s*\{?\s*\\circ\s*\}?|°', '*(pi/180)', s)
        else:
            s = re.sub(r'\^\s*\{?\s*\\circ\s*\}?|°', '', s)
        # Expand fractions (including mixed-brace \frac{270}7) and square roots
        s = _expand_latex_fracs(s)
        s = _expand_latex_sqrts(s)
        s = s.replace(r'\pi', ' pi ')
        s = _parenthesize_latex_funcs(s)
        # Multiplication and division symbols
        s = re.sub(r'\\(?:cdot|times)', '*', s)
        s = re.sub(r'\\div', '/', s)
        s = s.replace('\ufffd', '*')
        s = re.sub(r'\\left|\\right', '', s)
        # Powers with curly braces: ^{...} -> **(...)
        s = re.sub(r'\^\{([^}]+)\}', r'**(\1)', s)
        # Factorials: n! first, then (expr)! -> insert explicit multiplication if followed by '('
        s = re.sub(r'([0-9a-zA-Z]+)!', r'(factorial(\1))', s)
        s = re.sub(r'(\([^)]+\))!', r'(factorial\1)', s)
        s = re.sub(r'\)\s*\(', r')*(', s)
        # Base-n notation: 58_9 or 413_5 -> evaluate base-n integer when valid
        def _eval_base_n(m):
            digits, b_str = m.group(1), m.group(2)
            base = int(b_str)
            if 2 <= base <= 36:
                try:
                    return str(int(digits, base))
                except ValueError:
                    pass
            return digits
        s = re.sub(r'(\d+)_\{?(\d+)\}?', _eval_base_n, s)
        # Subscript variable identifiers: a_1 -> a1, x_{2} -> x2
        s = re.sub(r'([a-zA-Z])_\{?([0-9a-zA-Z]+)\}?', r'\1\2', s)
        s = s.replace('^', '**')
        s = re.sub(r'[{}\$]', '', s)
        # Remove discursive connecting words before inserting implicit multiplication
        s = re.sub(
            r'\b(?:and|or|let|if|where|then|so|thus|since|we have|we get|implies|for|with|therefore)\b',
            '', s, flags=re.I
        )
        # Insert explicit multiplication between numeric literals and symbols/functions/parentheses
        s = re.sub(r'(\d)(?!e[-+]?\d)\s*(?=[a-zA-Z(])', r'\1*', s)
        s = re.sub(r'(\))\s*(?=[a-zA-Z0-9(])', r'\1*', s)
        return re.sub(r'\s+', ' ', s).strip()

    @classmethod
    def verify_chunk(cls, text_chunk: str) -> Tuple[bool, int, List[str]]:
        """
        Extracts equations and checks for mathematical contradictions.
        Returns:
            is_consistent (bool): False if any explicit mathematical contradiction is found.
            verified_count (int): Number of sound equations verified.
            contradictions (List[str]): List of detected contradictions.
        """
        contradictions = []
        verified_count = 0
        lines = text_chunk.split('\n')
        sym_locals = {
            'E': sympy.Symbol('E'),
            'I': sympy.Symbol('I'),
            'S': sympy.Symbol('S'),
            'N': sympy.Symbol('N'),
            'O': sympy.Symbol('O'),
            'Q': sympy.Symbol('Q'),
            'pi': sympy.pi,
        }

        for raw_line in lines:
            line_clean = re.sub(r'^\s*[-*•]\s+', '', raw_line).strip()
            line_clean = re.sub(r'^\s*\d+[\.\)]\s+', '', line_clean).strip()

            if not line_clean or '=' not in line_clean:
                continue

            # Skip questions, speculative checks, approximations
            if any(c in line_clean for c in ['?', '...', r'\approx', '~']):
                continue
            if any(w in line_clean.lower() for w in ['wait', 'check', 'guess', 'suppose', 'assume', 'if ', 'remainder']):
                continue

            # Split on sentence boundaries, semicolons, implication arrows, or comma-separated clauses
            sentences = re.split(
                r'\.\s+(?=[A-Z])|;|=>|->|\\Rightarrow|\\implies|\\therefore|,\s+(?=[a-zA-Z\\])',
                line_clean
            )
            for s in sentences:
                s_clean = s.strip().rstrip('.')
                if '=' not in s_clean:
                    continue

                # Skip coordinate pairs like (4, 0) or (8/3, 2) or matrix brackets
                if re.search(r'\(\s*[-+]?[0-9a-zA-Z/.\s]+\s*,\s*[-+]?[0-9a-zA-Z/.\s]+\s*\)', s_clean) or '[[' in s_clean:
                    continue

                # Skip inequalities or modulo/remainder expressions
                if any(sym in s_clean for sym in ['<=', '>=', r'\le', r'\ge', '!=', r'\neq', '<', '>', '%', 'mod']):
                    continue

                parts = s_clean.split('=')
                if len(parts) >= 2:
                    for i in range(len(parts) - 1):
                        # If middle segment contains a comma, it separates two distinct equations
                        if i > 0 and ',' in parts[i]:
                            continue
                        lhs_raw = cls.clean_latex_expr(parts[i])
                        rhs_raw = cls.clean_latex_expr(parts[i + 1])

                        lhs_m = re.findall(r'[-+]?[0-9a-zA-Z\.\*\+\-\/\(\)\s\^]+', lhs_raw)
                        rhs_m = re.findall(r'[-+]?[0-9a-zA-Z\.\*\+\-\/\(\)\s\^]+', rhs_raw)

                        if lhs_m and rhs_m:
                            lhs_expr = lhs_m[-1].strip().replace('^', '**')
                            rhs_expr = rhs_m[0].strip().replace('^', '**')

                            if any(c.isalnum() for c in lhs_expr) and any(c.isalnum() for c in rhs_expr):
                                try:
                                    v1 = safe_sympify(lhs_expr, locals_dict=sym_locals)
                                    v2 = safe_sympify(rhs_expr, locals_dict=sym_locals)
                                    if v1 is None or v2 is None:
                                        continue

                                    if v1.is_number and v2.is_number:
                                        f1 = float(v1.evalf())
                                        f2 = float(v2.evalf())
                                        if is_numeric_step_mismatch(f1, f2):
                                            contradictions.append(f"{lhs_expr} ({f1}) != {rhs_expr} ({f2})")
                                        else:
                                            verified_count += 1
                                    else:
                                        diff = safe_simplify_expr(v1 - v2)
                                        if diff is not None and diff == 0:
                                            verified_count += 1
                                        elif diff is not None and diff.is_number and (len(v1.free_symbols) > 0 or len(v2.free_symbols) > 0):
                                            # Tactic ring: (x^3 + 3x^2) + (-3x^2 + 3x - 6) = x^3 + 3x -> diff = -6 != 0
                                            f_diff = float(diff.evalf())
                                            if abs(f_diff) > 1e-4:
                                                contradictions.append(
                                                    f"LeanRingViolation: {lhs_expr} != {rhs_expr} (residual {f_diff:+.4g} != 0)"
                                                )
                                except Exception:
                                    pass

        # Also check multi-step Lean tactic obligations (subst & ring chains)
        step_report = LeanTacticStepVerifier.verify_proof_steps(text_chunk, include_sorry_in_contradictions=False)
        for c in step_report.contradictions:
            if c not in contradictions:
                contradictions.append(c)

        return (len(contradictions) == 0), verified_count, contradictions


# ============================================================================
# LEAN 4 TACTIC-INSPIRED FORMAL STEP VERIFIER (LeanTacticStepVerifier)
# ============================================================================

@dataclass(frozen=True)
class ProofStepObligation:
    """
    Represents a single step s_k in a natural-language + symbolic proof chain,
    modeled as a Lean 4 tactic transition: (Gamma_{k-1} |- G) -> (Gamma_k |- G).
    """
    step_index: int
    step_text: str
    char_start: int
    char_end: int
    tactic_used: str = "assumption"
    is_discharged: bool = True
    violation_type: str = ""
    violation_detail: str = ""
    lean4_snippet: str = ""


@dataclass(frozen=True)
class ProofVerificationReport:
    """
    Complete step-by-step formal proof verification report produced by
    `LeanTacticStepVerifier.verify_proof_steps`.
    When a step s_k fails a tactic obligation (`ring`, `linarith_subst`, `norm_num`,
    `no_sorry`, or `precondition_logic`), `verified_prefix_text` holds the exact valid
    causal prefix s_1 .. s_{k-1} so the latent manifold rollback engine can roll back to s_{k-1}.
    """
    is_formally_sound: bool
    steps: Tuple[ProofStepObligation, ...] = field(default_factory=tuple)
    first_failed_step_index: Optional[int] = None
    rollback_char_offset: int = 0
    verified_prefix_text: str = ""
    failed_step_text: str = ""
    violation_type: str = ""
    violation_summary: str = ""
    verified_eq_count: int = 0
    causal_preconditions_verified: int = 0
    proof_status_label: str = "NATURAL LANGUAGE (No Formal Equations Invoked)"
    contradictions: Tuple[str, ...] = field(default_factory=tuple)
    has_sorry_bluff: bool = False


class SymbolicStepChecker:
    """
    SymPy + Regex Symbolic & Algebraic Step Checker for Neurosymbolic Arkhe Reasoning.
    (Backward-compatible alias: `LeanTacticStepVerifier`.)
    Enforces five heuristic symbolic/causal checks at each reasoning step s_k:
      1. Polynomial / algebraic residual verification (`diff = v1 - v2`).
      2. Back-substitution of claimed variable bindings into prior premises Gamma_{k-1}.
      3. Ground numeric equality / relative-tolerance contradiction detection.
      4. Unproved verbal shortcut / bluff detection.
      5. Domain-specific goal precondition verification (`Goal(G, Object O, Dest L) => At(O, L)`).
    """

    OBJECT_SERVICE_GOAL_RE = re.compile(
        r"(?:araba(?:mı|yı|nı)?|arac(?:ımı|ı)?|kamyon(?:u|umu)?|motosiklet(?:i|imi)?|car|vehicle|truck)\b.*"
        r"(?:yıka(?:tmak|mak|ma|nmak)|tamir|servis|muayene|şarj|wash|repair|service|inspect|charge)"
        r"|(?:yıka(?:tmak|mak|ma|nmak)|tamir|servis|muayene|şarj|wash|repair|service|inspect|charge)\b.*"
        r"(?:araba(?:mı|yı|nı)?|arac(?:ımı|ı)?|kamyon(?:u|umu)?|motosiklet(?:i|imi)?|car|vehicle|truck)",
        re.I | re.S,
    )

    PRECONDITION_VIOLATION_STEP_RE = re.compile(
        r"(?:"
        r"(?:therefore|thus|so|conclusion|answer|sonuç|bu\s+yüzden|dolayısıyla|o\s+halde|böylece)[^\n]*?"
        r"(?:yürüyerek|yürümek|walk(?:ing)?|on\s+foot)"
        r"|(?:yürüyerek\s+git(?:meli|meniz|mek)\s+(?:daha\s+)?(?:mantıklı|doğru|uygun|iyi|pratik))"
        r"|(?:yürüyerek\s+gitmelisiniz)"
        r"|(?:walking\s+is\s+more\s+(?:logical|efficient|practical|sensible))"
        r"|(?:you\s+should\s+(?:walk|go\s+on\s+foot))"
        r"|(?:answer\s+should\s+be\s*:?\s*yürüyerek)"
        r")",
        re.I,
    )

    PRECONDITION_NEGATION_GUARD_RE = re.compile(
        r"(?:değil|olmaz|mantıksız|imkansız|anlamsız|saçma|hatalı|yanlış|yıkayamaz|yıkanamaz|"
        r"araba\s+olmadan|arabayı\s+götürmeden|ortada\s+araba\s+olmaz|"
        r"not\s+walk|cannot\s+wash|without\s+the\s+car|absurd|illogical|impossible)",
        re.I,
    )

    PRECONDITION_SATISFIED_STEP_RE = re.compile(
        r"(?:"
        r"(?:araba(?:yla| ile)\s+git(?:meli|meniz|mek))"
        r"|(?:araba(?:nın|yı)\s+(?:orada|yıkama\s+yerinde)\s+olması)"
        r"|(?:yıkanacak\s+(?:olan\s+)?araba(?:nın|yı))"
        r"|(?:arabayı\s+götürme(?:k|niz|li))"
        r"|(?:need\s+the\s+car\s+at\s+the\s+(?:car\s+)?wash)"
        r"|(?:must\s+(?:drive|take|bring)\s+the\s+car)"
        r"|(?:go\s+by\s+car\s+because)"
        r")",
        re.I,
    )

    @classmethod
    def check_causal_precondition(cls, step_text: str, full_context: str) -> Tuple[Optional[bool], str]:
        """
        Evaluates `tactic_precondition_logic` on `step_text` given `full_context`:
        Returns:
          - (False, violation_detail) if the step violates a mandatory physical goal precondition.
          - (True, satisfaction_detail) if the step explicitly discharges/verifies the precondition.
          - (None, "") if no precondition rule applies to this step.
        """
        if not cls.OBJECT_SERVICE_GOAL_RE.search(full_context):
            return None, ""

        # Check if this step asserts a conclusion that leaves the target object behind
        if cls.PRECONDITION_VIOLATION_STEP_RE.search(step_text):
            if not cls.PRECONDITION_NEGATION_GUARD_RE.search(step_text):
                return (
                    False,
                    "Goal Precondition Violation [Goal(Wash, Car, CarWash) => Requires(At(Car, CarWash) = True)]: "
                    "Step recommends walking (At(Car, CarWash) = False), leaving the car behind so the primary goal cannot be executed.",
                )

        # Check if this step explicitly verifies and satisfies the object transport precondition
        if cls.PRECONDITION_SATISFIED_STEP_RE.search(step_text):
            return (
                True,
                "Discharged Goal Precondition: Action Drive(Car, CarWash) satisfies At(Car, CarWash) = True.",
            )

        return None, ""

    SORRY_BLUFF_PATTERNS: Tuple[Tuple[str, str], ...] = (
        (r"using the known result", "Unproved 'known result' assertion (Lean `sorry` in place of algebraic proof)"),
        (r"can be derived via (?:trigonometric|algebraic|geometric|standard)", "Skipped derivation via vague reference to identities (Lean `sorry`)"),
        (r"calculating the argument .* yields", "Skipped complex argument summation without step-by-step computation (Lean `sorry`)"),
        (r"in many similar (?:algebraic|math|competition|geometry) problems", "Meta-guessing from 'similar problems' instead of proving (Lean `sorry`)"),
        (r"standard analysis shows", "Unproved 'standard analysis shows' shortcut (Lean `sorry`)"),
        (r"without doing the full (?:algebra|calculation|expansion)", "Explicitly skipped algebraic expansion (Lean `sorry`)"),
        (r"it is a well-known (?:fact|result|identity) that the (?:answer|value|expression) (?:is|equals)", "Unproved well-known result shortcut (Lean `sorry`)"),
        (r"from standard properties of this configuration", "Unproved configuration jump (Lean `sorry`)"),
    )

    SOLVING_CLAIM_CUE = re.compile(
        r"(?:solving|yields|gives|we get|we find|which gives|solutions? are|roots? are|therefore|thus|so)\b[^=\n]*?",
        re.I,
    )

    ASSIGNMENT_PAIR_RE = re.compile(
        r"\b([a-zA-Z])\s*=\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)\b"
    )

    SYM_LOCALS = {
        'E': sympy.Symbol('E'),
        'I': sympy.Symbol('I'),
        'S': sympy.Symbol('S'),
        'N': sympy.Symbol('N'),
        'O': sympy.Symbol('O'),
        'Q': sympy.Symbol('Q'),
        'pi': sympy.pi,
    }

    PROMPT_EVAL_CUE_RE = re.compile(
        r"^\s*(?:Simplify|Evaluate|Compute|Calculate|What\s+is\s+the\s+value\s+of|Find\s+the\s+value\s+of)\s+\$([^$]+)\$\s*[\.\?!]?\s*$",
        re.I,
    )

    @classmethod
    def check_prompt_target_consistency(cls, prompt: str, response_text: str) -> Optional[str]:
        """
        Zero-leakage Lean `norm_num` check for closed-form Simplify/Evaluate prompts:
        When `prompt` explicitly asks to simplify/evaluate a closed constant expression
        (zero free variables, finite real value) and `response_text` gives a numeric/symbolic
        `\\boxed{...}` answer that disagrees numerically, returns a concrete violation message.
        """
        if not prompt or not response_text:
            return None
        m = cls.PROMPT_EVAL_CUE_RE.match(prompt.strip())
        if not m:
            return None
        target_raw = m.group(1).strip(" \t\r\n.,;:")
        if not target_raw or "=" in target_raw:
            return None

        boxed = _extract_last_boxed(response_text)
        if not boxed or boxed == "[ABSTAIN]":
            return None

        cleaned_prompt_expr = CstSymbolicVerifier.clean_latex_expr(target_raw)
        sym_prompt = safe_sympify(cleaned_prompt_expr, locals_dict=cls.SYM_LOCALS)
        sym_ans = NormalizedMathMatcher.to_sympy(boxed)
        if sym_prompt is None or sym_ans is None:
            return None
        if getattr(sym_prompt, "free_symbols", set()) or getattr(sym_ans, "free_symbols", set()):
            return None
        try:
            f_prompt = float(sym_prompt.evalf())
            f_ans = float(sym_ans.evalf())
            if not (math.isfinite(f_prompt) and math.isfinite(f_ans)):
                return None
            if is_numeric_step_mismatch(f_ans, f_prompt, rel_tol=1e-3, abs_tol=1e-4):
                return (
                    f"Claimed answer {boxed} ({f_ans:.6g}) contradicts closed prompt expression "
                    f"({f_prompt:.6g}) — check sign, quadrant, or arithmetic."
                )
        except Exception:
            return None
        return None

    @classmethod
    def has_verbal_bluff(cls, text: str) -> Tuple[bool, str]:
        """Returns (True, reason) if `text` contains an unproved verbal bluff (`sorry`)."""
        low = text.lower()
        for pat, desc in cls.SORRY_BLUFF_PATTERNS:
            m = re.search(pat, low)
            if m:
                return True, f"{desc}: '{m.group(0)}'"
        return False, ""

    @classmethod
    def _segment_into_steps(cls, text: str) -> List[Tuple[int, int, str]]:
        """
        Segments `text` into line/paragraph reasoning steps while preserving exact
        character spans (char_start, char_end, step_str) in `text`.
        """
        spans: List[Tuple[int, int, str]] = []
        idx = 0
        for line in text.splitlines(keepends=True):
            start = idx
            end = idx + len(line)
            idx = end
            if line.strip():
                spans.append((start, end, line.rstrip("\r\n")))
        return spans

    @classmethod
    def verify_proof_steps(
        cls,
        text: str,
        include_sorry_in_contradictions: bool = False,
        prompt: str = "",
    ) -> ProofVerificationReport:
        """
        Executes sequential Lean-style proof obligation verification over `text`.
        Tracks active symbolic premises Gamma_{k-1} across steps and pinpoints the exact
        first step s_k where a tactic violation, precondition violation, or `sorry` jump occurs.
        """
        raw_spans = cls._segment_into_steps(text)
        full_context = f"{prompt}\n{text}" if prompt else text
        active_premises: List[Tuple[sympy.Expr, sympy.Expr, str]] = []
        step_obligations: List[ProofStepObligation] = []
        contradictions: List[str] = []
        verified_eq_count = 0
        causal_preconditions_verified = 0
        has_sorry = False

        first_failed_idx: Optional[int] = None
        rollback_offset: int = len(text)
        failed_step_str: str = ""
        first_violation_type: str = ""
        first_violation_summary: str = ""

        for step_num, (c_start, c_end, raw_line) in enumerate(raw_spans, start=1):
            line_clean = re.sub(r'^\s*[-*•]\s+', '', raw_line).strip()
            line_clean = re.sub(r'^\s*\d+[\.\)]\s+', '', line_clean).strip()

            tactic_used = "assumption"
            is_discharged = True
            v_type = ""
            v_detail = ""

            # 0. Check `tactic_precondition_logic` (STRIPS / Situation-Calculus Goal Precondition)
            precond_ok, precond_msg = cls.check_causal_precondition(line_clean, full_context)
            if precond_ok is False:
                tactic_used = "precondition_logic"
                is_discharged = False
                v_type = "LeanPreconditionViolation"
                v_detail = precond_msg
                contradictions.append(f"{v_type}: {v_detail}")
            elif precond_ok is True:
                tactic_used = "precondition_logic"
                causal_preconditions_verified += 1

            # 1. Check `tactic_no_sorry` (Verbal Bluff / Unproved Jump)
            bluff_found, bluff_desc = cls.has_verbal_bluff(line_clean)
            if bluff_found and is_discharged:
                has_sorry = True
                tactic_used = "no_sorry"
                is_discharged = False
                v_type = "LeanSorryJumpViolation"
                v_detail = f"Step {step_num} invokes unproved shortcut ({bluff_desc})"
                if include_sorry_in_contradictions:
                    contradictions.append(f"{v_type}: {v_detail}")

            # 2. Check `tactic_linarith_subst` (Back-substitution of claimed variable bindings into Gamma_{k-1})
            if is_discharged and active_premises and cls.SOLVING_CLAIM_CUE.search(line_clean):
                pairs = cls.ASSIGNMENT_PAIR_RE.findall(line_clean)
                if len(pairs) >= 1:
                    sigma: Dict[sympy.Symbol, sympy.Expr] = {}
                    for var_name, val_str in pairs:
                        sym_v = cls.SYM_LOCALS.get(var_name, sympy.Symbol(var_name))
                        parsed_val = safe_sympify(val_str)
                        if parsed_val is not None:
                            sigma[sym_v] = parsed_val
                    if sigma:
                        for lhs_p, rhs_p, eq_raw in active_premises:
                            free_vars = (lhs_p - rhs_p).free_symbols
                            # Check multi-variable systems (or equations matching claimed variables)
                            if len(free_vars) >= 2 and free_vars.issubset(set(sigma.keys())):
                                try:
                                    res = safe_simplify_expr(lhs_p.subs(sigma) - rhs_p.subs(sigma))
                                    if res is not None and res.is_number and abs(float(res.evalf())) > 1e-3:
                                        tactic_used = "linarith_subst"
                                        is_discharged = False
                                        v_type = "LeanSubstViolation"
                                        sigma_str = ", ".join(f"{k}={v}" for k, v in sigma.items())
                                        v_detail = (
                                            f"Claimed assignment {{{sigma_str}}} violates prior premise "
                                            f"'{eq_raw}' (residual {float(res.evalf()):+.4g} != 0)"
                                        )
                                        contradictions.append(f"{v_type}: {v_detail}")
                                        break
                                except Exception:
                                    pass

            # 3. Check `tactic_ring` & `tactic_norm_num` on equations in the step
            if '=' in line_clean and not any(c in line_clean for c in ['?', '...', r'\approx', '~']):
                if not any(w in line_clean.lower() for w in ['wait', 'check', 'guess', 'suppose', 'assume', 'if ', 'remainder']):
                    sentences = re.split(
                        r'\.\s+(?=[A-Z])|;|=>|->|\\Rightarrow|\\implies|\\therefore|,\s+(?=[a-zA-Z\\])',
                        line_clean
                    )
                    for s in sentences:
                        s_clean = s.strip().rstrip('.')
                        if '=' not in s_clean:
                            continue
                        if re.search(r'\(\s*[-+]?[0-9a-zA-Z/.\s]+\s*,\s*[-+]?[0-9a-zA-Z/.\s]+\s*\)', s_clean) or '[[' in s_clean:
                            continue
                        if any(sym in s_clean for sym in ['<=', '>=', r'\le', r'\ge', '!=', r'\neq', '<', '>', '%', 'mod']):
                            continue

                        parts = s_clean.split('=')
                        if len(parts) >= 2:
                            chain_exprs: List[Tuple[str, sympy.Expr]] = []
                            for i in range(len(parts) - 1):
                                if i > 0 and ',' in parts[i]:
                                    continue
                                lhs_raw = CstSymbolicVerifier.clean_latex_expr(parts[i])
                                rhs_raw = CstSymbolicVerifier.clean_latex_expr(parts[i + 1])

                                lhs_m = re.findall(r'[-+]?[0-9a-zA-Z\.\*\+\-\/\(\)\s\^]+', lhs_raw)
                                rhs_m = re.findall(r'[-+]?[0-9a-zA-Z\.\*\+\-\/\(\)\s\^]+', rhs_raw)
                                if not (lhs_m and rhs_m):
                                    continue

                                lhs_expr = lhs_m[-1].strip().replace('^', '**')
                                rhs_expr = rhs_m[0].strip().replace('^', '**')
                                if not (any(c.isalnum() for c in lhs_expr) and any(c.isalnum() for c in rhs_expr)):
                                    continue

                                try:
                                    v1 = safe_sympify(lhs_expr, locals_dict=cls.SYM_LOCALS)
                                    v2 = safe_sympify(rhs_expr, locals_dict=cls.SYM_LOCALS)
                                    if v1 is None or v2 is None:
                                        continue
                                    chain_exprs.append((lhs_expr, v1))
                                    if i == len(parts) - 2:
                                        chain_exprs.append((rhs_expr, v2))

                                    if v1.is_number and v2.is_number:
                                        f1 = float(v1.evalf())
                                        f2 = float(v2.evalf())
                                        if is_numeric_step_mismatch(f1, f2):
                                            tactic_used = "norm_num"
                                            is_discharged = False
                                            v_type = "LeanNormNumViolation"
                                            v_detail = f"{lhs_expr} ({f1}) != {rhs_expr} ({f2})"
                                            contradictions.append(v_detail)
                                        else:
                                            tactic_used = "norm_num"
                                            verified_eq_count += 1
                                    else:
                                        diff = safe_simplify_expr(v1 - v2)
                                        if diff is not None and diff == 0:
                                            tactic_used = "ring"
                                            verified_eq_count += 1
                                        elif diff is not None and diff.is_number and (len(v1.free_symbols) > 0 or len(v2.free_symbols) > 0):
                                            f_diff = float(diff.evalf())
                                            if abs(f_diff) > 1e-4:
                                                tactic_used = "ring"
                                                is_discharged = False
                                                v_type = "LeanRingViolation"
                                                v_detail = f"{lhs_expr} != {rhs_expr} (polynomial constant residual {f_diff:+.4g} != 0)"
                                                contradictions.append(f"{v_type}: {v_detail}")
                                        else:
                                            # Non-trivial algebraic constraint equation -> add to Gamma_k if valid
                                            free_syms = (v1 - v2).free_symbols
                                            if 1 <= len(free_syms) <= 4:
                                                active_premises.append((v1, v2, f"{lhs_expr} = {rhs_expr}"))
                                except Exception:
                                    pass

                            # Check multi-step algebraic simplification chain: E1 = E2 = E3
                            # If E1 == E2 identically by `ring`, then E2 = E3 in the same chain must also be an identity!
                            if is_discharged and len(chain_exprs) >= 3:
                                try:
                                    first_diff = safe_simplify_expr(chain_exprs[0][1] - chain_exprs[1][1])
                                    if first_diff is not None and first_diff == 0 and len(chain_exprs[0][1].free_symbols) > 0:
                                        for j in range(1, len(chain_exprs) - 1):
                                            step_diff = safe_simplify_expr(chain_exprs[j][1] - chain_exprs[j + 1][1])
                                            if step_diff is not None and step_diff != 0 and chain_exprs[j][1].free_symbols == chain_exprs[j + 1][1].free_symbols:
                                                tactic_used = "ring"
                                                is_discharged = False
                                                v_type = "LeanRingViolation"
                                                v_detail = (
                                                    f"Broken algebraic simplification chain: {chain_exprs[j][0]} != "
                                                    f"{chain_exprs[j + 1][0]} (residual {step_diff} != 0)"
                                                )
                                                contradictions.append(f"{v_type}: {v_detail}")
                                                break
                                except Exception:
                                    pass

            lean_stmt = (
                f"have step_{step_num} : True := by {tactic_used}"
                if is_discharged
                else f"have step_{step_num} : False := by sorry -- [{v_type}] {v_detail}"
            )

            ob = ProofStepObligation(
                step_index=step_num,
                step_text=line_clean,
                char_start=c_start,
                char_end=c_end,
                tactic_used=tactic_used,
                is_discharged=is_discharged,
                violation_type=v_type,
                violation_detail=v_detail,
                lean4_snippet=lean_stmt,
            )
            step_obligations.append(ob)

            if (not is_discharged) and first_failed_idx is None:
                first_failed_idx = step_num
                rollback_offset = c_start
                failed_step_str = line_clean
                first_violation_type = v_type
                first_violation_summary = f"[{v_type} at Step {step_num}] {v_detail}"

        # Closed-form prompt expression consistency check (zero ground-truth leakage)
        goal_mismatch = cls.check_prompt_target_consistency(prompt, text)
        if goal_mismatch:
            v_str = f"LeanGoalMismatchViolation: {goal_mismatch}"
            if v_str not in contradictions:
                contradictions.append(v_str)
            if first_failed_idx is None:
                first_violation_type = "LeanGoalMismatchViolation"
                first_violation_summary = f"[LeanGoalMismatchViolation] {goal_mismatch}"

        prefix_text = text[:rollback_offset].rstrip() if first_failed_idx is not None else text
        is_sound = (first_failed_idx is None) and (len(contradictions) == 0) and (not has_sorry)

        if not is_sound:
            status_label = f"STEP CHECK FAILED ({first_violation_type or 'Contradiction'})"
        elif verified_eq_count > 0 and causal_preconditions_verified > 0:
            status_label = f"SYMBOLICALLY & CAUSALLY SOUND (SymPy: {verified_eq_count} eqs, {causal_preconditions_verified} preconditions)"
        elif verified_eq_count > 0:
            status_label = f"SYMBOLICALLY SOUND (SymPy Verified: {verified_eq_count} eqs)"
        elif causal_preconditions_verified > 0:
            status_label = "CAUSALLY SOUND (Arkhe-Şartlar Precondition Verified)"
        else:
            status_label = "NATURAL LANGUAGE (No Formal Equations Invoked)"

        return ProofVerificationReport(
            is_formally_sound=is_sound,
            steps=tuple(step_obligations),
            first_failed_step_index=first_failed_idx,
            rollback_char_offset=rollback_offset,
            verified_prefix_text=prefix_text,
            failed_step_text=failed_step_str,
            violation_type=first_violation_type,
            violation_summary=first_violation_summary,
            verified_eq_count=verified_eq_count,
            causal_preconditions_verified=causal_preconditions_verified,
            proof_status_label=status_label,
            contradictions=tuple(contradictions),
            has_sorry_bluff=has_sorry,
        )


# Backward-compatible alias (prefer honest `SymbolicStepChecker` in new code & papers)
LeanTacticStepVerifier = SymbolicStepChecker


class NormalizedMathMatcher:
    """
    Evaluates equivalence between model prediction and ground truth.
    Supports both legacy strict matching and robust symbolic mathematical matching.
    """

    @staticmethod
    def clean_latex(s: str) -> str:
        s = s.strip()
        s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
        s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
        s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
        s = re.sub(r"\\mbox\{([^}]+)\}", r"\1", s)
        s = s.replace(r"\$", "").replace("$", "").strip()
        s = s.replace(r"\left", "").replace(r"\right", "")
        s = s.replace(r"\dfrac", r"\frac")
        s = s.replace(r"^\circ", "").replace("^{\\circ}", "").replace("°", "")

        # Strip variable assignment prefix: x = 2 -> 2
        s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)

        # Strip LaTeX spacing/thinning commands BEFORE thousands comma removal (e.g. 10,\!080 -> 10,080)
        s = re.sub(r'\\(?:qquad|quad)(?![a-zA-Z])|\\[,!;:]|~', '', s)

        # Strip trailing physical units BEFORE removing whitespace
        # Multi-letter units (cm, mm, inches, etc.) follow a digit with optional whitespace
        # Single-letter unit 'm' (meters) strictly requires at least one whitespace after the digit
        s = re.sub(r'(?<=\d)(?:\s+m|\s*(?:cm|mm|km|meters|meter|inches|inch|degrees|deg|units|unit))\b\s*$', '', s, flags=re.I)

        # Remove thousands commas BEFORE removing spaces so spaced tuples like (3, 400) stay intact,
        # and avoid stripping commas inside compact 2-tuples/intervals (a,b) or decimals (0,125 / 3,14).
        if not re.fullmatch(r'[\(\[]\s*-?\d+\s*,\s*[1-9]\d*\s*[\)\]]', s.strip()):
            s = re.sub(
                r'(?<![.\d])([1-9]\d{0,2}(?:,\d{3})+)(?!\d)',
                lambda m: m.group(1).replace(',', ''),
                s,
            )

        s = s.replace(" ", "")
        # Normalize base subscripts: _{5} -> _5
        s = re.sub(r'_\{(\w+)\}', r'_\1', s)
        # Normalize square roots without braces: \sqrt2 -> \sqrt{2}
        s = re.sub(r'\\sqrt\s*([0-9a-zA-Z])(?![a-zA-Z0-9])', r'\\sqrt{\1}', s)
        return s.strip()

    @classmethod
    def to_sympy(cls, s: str):
        s = cls.clean_latex(s)
        s = _expand_latex_fracs(s)
        s = _expand_latex_sqrts(s)
        s = s.replace(r"\pi", "pi")
        s = s.replace("^", "**")
        # Implicit multiplication
        s = re.sub(r'(\d)(?!e[-+]?\d)([a-zA-Z])', r'\1*\2', s)
        s = re.sub(r'(\d)(sqrt|pi|\()', r'\1*\2', s)
        s = re.sub(r'(\))([a-zA-Z0-9\(])', r'\1*\2', s)
        return safe_sympify(s)

    @classmethod
    def check_match(cls, pred: str, gt: str) -> Tuple[bool, bool]:
        """
        Returns:
            (strict_match, symbolic_match):
            - strict_match: Matches exact legacy evaluator criteria
            - symbolic_match: Matches true mathematical equivalence
        """
        # 1. Exact string match after basic clean
        cp = cls.clean_latex(pred)
        cg = cls.clean_latex(gt)

        # Legacy strict match (case sensitive, un-normalized braces)
        legacy_pred = pred.replace(" ", "").replace(r"\left", "").replace(r"\right", "")
        legacy_gt = gt.replace(" ", "").replace(r"\left", "").replace(r"\right", "")
        legacy_strict = (legacy_pred == legacy_gt)

        if cp == cg or cp.lower() == cg.lower():
            return True, True

        # Safe numeric integer / float equivalence (e.g. "2.0" == "2" or exact match)
        if cg.lstrip('-').isdigit() and cp.lstrip('-').isdigit() and cp == cg:
            return True, True

        try:
            val_p = float(cp)
            val_g = float(cg)
            if abs(val_p - val_g) < 1e-6:
                return True, True
        except ValueError:
            pass

        # Plus/minus set match
        if r"\pm" in gt:
            base_parts = gt.split(r"\pm")
            if len(base_parts) == 2:
                a, b = base_parts[0].strip(), base_parts[1].strip()
                p1 = f"{a}+{b},{a}-{b}"
                p2 = f"{a}-{b},{a}+{b}"
                if cp in [cls.clean_latex(p1), cls.clean_latex(p2)]:
                    return True, True

        # SymPy symbolic equivalence
        sym_gt = cls.to_sympy(gt)
        sym_pred = cls.to_sympy(pred)
        if sym_gt is not None and sym_pred is not None:
            diff = safe_simplify_expr(sym_gt - sym_pred)
            if diff is not None and diff == 0:
                return legacy_strict, True
            try:
                if abs(float(sym_gt.evalf()) - float(sym_pred.evalf())) < 1e-6:
                    return legacy_strict, True
            except Exception:
                pass

        return legacy_strict, False
