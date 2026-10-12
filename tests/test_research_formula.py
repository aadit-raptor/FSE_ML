"""The formula language of a sector model (PLAN.md R4 part b): reading,
working out and writing as Excel text."""
import random

import pytest

from research import formula
from research.formula import Bin, Call, DivisionByNil, FormulaError, Neg, Num, Ref
from tests.research_support import _Excel

NAMES = {"a": 7.0, "b": 2.0, "c": 5.0, "nil": 0.0}


def worked(text, names=NAMES):
    return formula.value(text, names)


@pytest.mark.parametrize("text, answer", [
    ("2 + 3 * 4", 14), ("(2 + 3) * 4", 20), ("8 / 4 / 2", 1), ("10 - 4 - 3", 3), ("10 - (4 - 3)", 9),
    ("a - b * c", -3), ("-a * b", -14), ("a - -b", 9), ("0.5 * a", 3.5), (".5 * a", 3.5),
    ("max(a, b, c)", 7), ("min(a, b, c)", 2), ("max(a - 10, 0)", 0), ("min(a, b) * max(b, c)", 10),
    ("div(a, b)", 3.5), ("div(a, nil)", 0), ("div(a, b - 2)", 0), ("1 + div(a, nil) * 9", 1),
])
def test_arithmetic_is_worked_out_as_written(text, answer):
    assert worked(text) == pytest.approx(answer, abs=1e-12)


def test_a_plain_division_by_nil_is_an_error_and_div_is_not():
    with pytest.raises(DivisionByNil):
        worked("a / nil")
    with pytest.raises(DivisionByNil):
        worked("a / (b - 2)")
    assert worked("div(a / nil, nil)") == 0      # the top is never worked out when the bottom is nil


def test_a_name_may_point_at_another_period():
    tree = formula.parse("cash[prev] + rev - rev[ytd]")
    assert list(formula.refs(tree)) == [Ref("cash", "prev"), Ref("rev"), Ref("rev", "ytd")]


@pytest.mark.parametrize("text", [
    "", "   ", "a +", "a b", "2 ** 3", "a ^ 2", "(a", "a)", "a[", "a[1]", "a[prev", "max(a)", "div(a)", "div(a, b, c)",
    "sum(a, b)", "round(a, 2)", "__import__('os')", "a.b", "a; b", "a if b else c", "a == b", "a > b", "1e9", "a % b",
    "lambda: 1", "max()", "a,b", "[a]",
])
def test_anything_else_is_refused(text):
    with pytest.raises(FormulaError):
        formula.parse(text)


def test_a_formula_has_a_size():
    with pytest.raises(FormulaError):
        formula.parse("(" * 41 + "a" + ")" * 41)
    with pytest.raises(FormulaError):
        formula.parse("-" * 41 + "a")
    with pytest.raises(FormulaError):
        formula.parse("+".join(["a"] * 200))
    with pytest.raises(FormulaError):
        formula.parse("a" * 1001)
    assert worked("+".join(["a"] * 150)) == 150 * 7          # the largest allowed still works out


def test_value_refuses_a_name_it_was_not_given():
    with pytest.raises(FormulaError):
        worked("a + missing")
    with pytest.raises(FormulaError):
        worked("a[prev]")


CELLS = {"a": "A1", "b": "B1", "c": "C1", "nil": "D1"}


def excel(text):
    return formula.to_excel(formula.parse(text), lambda ref: CELLS[ref.name])


@pytest.mark.parametrize("text, written", [
    ("a + b * c", "A1+B1*C1"),
    ("(a + b) * c", "(A1+B1)*C1"),
    ("a - (b - c)", "A1-(B1-C1)"),
    ("a - b - c", "A1-B1-C1"),
    ("a / (b * c)", "A1/(B1*C1)"),
    ("a / b * c", "A1/B1*C1"),
    ("-a", "(0-A1)"),
    ("a * -b", "A1*(0-B1)"),
    ("-(a + b)", "(0-(A1+B1))"),
    ("max(a - b, 0)", "MAX(A1-B1,0)"),
    ("min(a, b, c)", "MIN(A1,B1,C1)"),
    ("div(a, b)", "IF(B1=0,0,A1/B1)"),
    ("div(a + b, b * c)", "IF(B1*C1=0,0,(A1+B1)/(B1*C1))"),
    ("0.064 * a", "0.064*A1"),
])
def test_excel_text(text, written):
    assert excel(text) == written


def _random_tree(rng, depth=0):
    roll = rng.random()
    if depth > 4 or roll < 0.25:
        return Ref(rng.choice("abc")) if rng.random() < 0.7 else Num(float(n := rng.choice([0, 1, 2, 0.5, 365])), repr(n))
    if roll < 0.35:
        return Neg(_random_tree(rng, depth + 1))
    if roll < 0.85:
        return Bin(rng.choice("+-*/"), _random_tree(rng, depth + 1), _random_tree(rng, depth + 1))
    fn = rng.choice(["max", "min", "div"])
    return Call(fn, tuple(_random_tree(rng, depth + 1) for _ in range(2 if fn == "div" else rng.choice([2, 3]))))


def _text(node):
    """A tree back in the formula language, fully bracketed."""
    if isinstance(node, Num):
        return node.text
    if isinstance(node, Ref):
        return node.name
    if isinstance(node, Neg):
        return "(-%s)" % _text(node.of)
    if isinstance(node, Bin):
        return "(%s %s %s)" % (_text(node.left), node.op, _text(node.right))
    return "%s(%s)" % (node.fn, ", ".join(_text(a) for a in node.args))


def test_excel_text_means_what_the_formula_means():
    """500 random formulas: the Excel text, read by a reader that shares nothing with
    this one, gives the same number, or both divide by nil."""
    rng = random.Random(4)
    cells = {"A1": 7.0, "B1": 2.0, "C1": -5.0}
    names = {"a": 7.0, "b": 2.0, "c": -5.0}
    compared = 0
    for _ in range(500):
        text = _text(_random_tree(rng))
        tree = formula.parse(text)
        written = formula.to_excel(tree, lambda ref: {"a": "A1", "b": "B1", "c": "C1"}[ref.name])
        try:
            ours = formula.compute(tree, lambda ref: names[ref.name])
        except DivisionByNil:
            with pytest.raises(ZeroDivisionError):
                _Excel(written, cells.__getitem__).expression()
            continue
        theirs = _Excel(written, cells.__getitem__).expression()
        assert theirs == pytest.approx(ours, rel=1e-12, abs=1e-12), (text, written)
        compared += 1
    assert compared > 300
