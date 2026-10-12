"""The formula language of a sector model (PLAN.md R4).

Plain arithmetic, as the reference cases use: ``+ - * /``, brackets, numbers,
``max``, ``min`` and ``div``. A name is a row in the same period or a
constant; ``cash[prev]`` is the period before and ``rev[ytd]`` a named
period. ``div(a, b)`` is a over b, or nil when b is nil: the one way a row
divides by something that may be switched off.

Two limits come from Excel. A formula's Excel text may not pass 8,192
characters (`div` writes its bottom twice, so nesting it doubles the text),
and `check` in `definition.py` refuses a formula that would. And Excel reads
a sum that comes out within rounding of nil as exactly nil, where this module
keeps the tiny number: `div(1, 0.1 + 0.2 - 0.3)` is nil in Excel and huge
here. No model row divides by such a difference; the LibreOffice test in CI
is where a real case would show.

A formula is read once into a small tree. The evaluator walks that tree and
the workbook writer turns the same tree into Excel text, so the two cannot
read a formula differently. Nothing here calls ``eval``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator, Union

FUNCTIONS = {"max": (2, None), "min": (2, None), "div": (2, 2)}   # name -> fewest, most arguments
MAX_LENGTH = 1000           # characters in one formula
MAX_TOKENS = 300            # numbers, names and signs in one formula: bounds how deep its tree can go
MAX_DEPTH = 40              # brackets inside brackets, or minus signs in a row
MAX_EXCEL = 8192            # characters Excel allows in one formula


class FormulaError(ValueError):
    """A formula that cannot be read. The message says where and why."""


class DivisionByNil(ArithmeticError):
    """A plain ``/`` whose bottom came out nil."""


@dataclass(frozen=True)
class Num:
    value: float
    text: str


@dataclass(frozen=True)
class Ref:
    name: str
    period: str | None = None      # None = the same period; "prev" = the period before


@dataclass(frozen=True)
class Neg:
    of: "Node"


@dataclass(frozen=True)
class Bin:
    op: str
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Call:
    fn: str
    args: tuple


Node = Union[Num, Ref, Neg, Bin, Call]

_TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?|\.\d+)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*/(),\[\]]))", re.ASCII)


def _tokens(text: str) -> list[tuple[str, str, int]]:
    out, at = [], 0
    while at < len(text):
        if text[at:].strip() == "":
            break
        m = _TOKEN.match(text, at)
        if m is None:
            raise FormulaError("cannot read %r at character %d of %r" % (text[at:].strip()[:1], at + 1, text))
        kind = m.lastgroup
        out.append((kind, m.group(kind), m.start(kind)))
        at = m.end()
    return out


class _Reader:
    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokens(text)
        if len(self.tokens) > MAX_TOKENS:
            raise FormulaError("a formula of more than %d parts: %r..." % (MAX_TOKENS, text[:40]))
        self.at = 0
        self.depth = 0

    def fail(self, why: str) -> FormulaError:
        return FormulaError("%s in %r" % (why, self.text))

    def peek(self) -> str | None:
        return self.tokens[self.at][1] if self.at < len(self.tokens) else None

    def take(self, want: str | None = None) -> tuple[str, str]:
        if self.at >= len(self.tokens):
            raise self.fail("the formula ends too soon")
        kind, value, _ = self.tokens[self.at]
        if want is not None and value != want:
            raise self.fail("expected %r, found %r" % (want, value))
        self.at += 1
        return kind, value

    def expression(self) -> Node:
        node = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()[1]
            node = Bin(op, node, self.term())
        return node

    def term(self) -> Node:
        node = self.unary()
        while self.peek() in ("*", "/"):
            op = self.take()[1]
            node = Bin(op, node, self.unary())
        return node

    def unary(self) -> Node:
        if self.peek() == "-":
            self.take()
            self.deeper()
            node = Neg(self.unary())
            self.depth -= 1
            return node
        return self.atom()

    def deeper(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise self.fail("brackets or minus signs nest deeper than %d" % MAX_DEPTH)

    def atom(self) -> Node:
        kind, value = self.take()
        if kind == "num":
            return Num(float(value), value)
        if kind == "name":
            if self.peek() == "(":
                return self.call(value)
            if self.peek() == "[":
                self.take()
                kind, period = self.take()
                if kind != "name":
                    raise self.fail("a period name belongs inside [ ]")
                self.take("]")
                return Ref(value, period)
            return Ref(value)
        if value == "(":
            self.deeper()
            node = self.expression()
            self.take(")")
            self.depth -= 1
            return node
        raise self.fail("unexpected %r" % value)

    def call(self, fn: str) -> Node:
        if fn not in FUNCTIONS:
            raise self.fail("unknown function %r (there are only max, min and div)" % fn)
        self.take("(")
        self.deeper()
        args = [self.expression()]
        while self.peek() == ",":
            self.take()
            args.append(self.expression())
        self.take(")")
        self.depth -= 1
        fewest, most = FUNCTIONS[fn]
        if len(args) < fewest or (most is not None and len(args) > most):
            raise self.fail("%s takes %s arguments, not %d" % (fn, fewest if most == fewest else "%d or more" % fewest, len(args)))
        return Call(fn, tuple(args))


def parse(text: str) -> Node:
    """Read a formula into its tree."""
    if not isinstance(text, str) or not text.strip():
        raise FormulaError("an empty formula")
    if len(text) > MAX_LENGTH:
        raise FormulaError("a formula longer than %d characters" % MAX_LENGTH)
    reader = _Reader(text)
    node = reader.expression()
    if reader.at != len(reader.tokens):
        raise reader.fail("unexpected %r" % reader.peek())
    return node


def refs(node: Node) -> Iterator[Ref]:
    """Every row or constant a formula names, in reading order."""
    if isinstance(node, Ref):
        yield node
    elif isinstance(node, Neg):
        yield from refs(node.of)
    elif isinstance(node, Bin):
        yield from refs(node.left)
        yield from refs(node.right)
    elif isinstance(node, Call):
        for arg in node.args:
            yield from refs(arg)


def compute(node: Node, lookup: Callable[[Ref], float]) -> float:
    """Work a tree out. `lookup` gives the value a name stands for."""
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Ref):
        return lookup(node)
    if isinstance(node, Neg):
        return -compute(node.of, lookup)
    if isinstance(node, Bin):
        left, right = compute(node.left, lookup), compute(node.right, lookup)
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if right == 0:
            raise DivisionByNil()
        return left / right
    if node.fn == "div":
        bottom = compute(node.args[1], lookup)
        return 0.0 if bottom == 0 else compute(node.args[0], lookup) / bottom
    values = [compute(arg, lookup) for arg in node.args]
    return max(values) if node.fn == "max" else min(values)


def value(text: str, names: dict) -> float:
    """A formula over plain names, worked out: `value("a - b", {"a": 3, "b": 1})`."""
    def lookup(ref: Ref) -> float:
        if ref.period is not None or ref.name not in names:
            raise FormulaError("unknown name %r in %r" % (ref.name if ref.period is None else "%s[%s]" % (ref.name, ref.period), text))
        return names[ref.name]
    return compute(parse(text), lookup)


_LEVEL = {"+": 1, "-": 1, "*": 2, "/": 2}


def to_excel(node: Node, address: Callable[[Ref], str]) -> str:
    """The same tree as Excel text (no leading =). `address` gives a name's cell."""
    return _excel(node, address, 0, False)


def _excel(node: Node, address: Callable[[Ref], str], outer: int, right_side: bool) -> str:
    text = _excel_text(node, address, outer, right_side)
    if len(text) > MAX_EXCEL:       # checked at every level, so a doubling formula stops early
        raise FormulaError("the formula is longer than the %d characters Excel allows once written out" % MAX_EXCEL)
    return text


def _excel_text(node: Node, address: Callable[[Ref], str], outer: int, right_side: bool) -> str:
    if isinstance(node, Num):
        return node.text
    if isinstance(node, Ref):
        return address(node)
    if isinstance(node, Neg):
        # "0 - x", in brackets: Excel then reads it as arithmetic wherever it stands
        return "(0-%s)" % _excel(node.of, address, 1, True)
    if isinstance(node, Bin):
        level = _LEVEL[node.op]
        text = "%s%s%s" % (_excel(node.left, address, level, False), node.op, _excel(node.right, address, level, True))
        # a - (b - c) and a / (b * c) keep their brackets; a + (b + c) is kept too, for the same sum order
        return "(%s)" % text if level < outer or (level == outer and right_side) else text
    if node.fn == "div":
        top = _excel(node.args[0], address, 2, False)
        bottom = _excel(node.args[1], address, 0, False)
        divisor = "(%s)" % bottom if isinstance(node.args[1], Bin) else bottom
        return "IF(%s=0,0,%s/%s)" % (bottom, top, divisor)
    args = [_excel(arg, address, 0, False) for arg in node.args]
    return "%s(%s)" % (node.fn.upper(), ",".join(args))
