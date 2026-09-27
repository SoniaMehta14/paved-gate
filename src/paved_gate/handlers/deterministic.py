"""Deterministic handlers for DIRECT_CODE_EXEC requests.

These do real work with no model in the loop and no dynamic code execution: every
computation goes through a hand-written parser or a lookup table.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import date
from decimal import Decimal, DivisionByZero, InvalidOperation, Overflow, localcontext

from pydantic import JsonValue

from paved_gate.handlers.base import DeterministicHandler

# ---------------------------------------------------------------- arithmetic


class ExpressionError(ValueError):
    pass


_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?|\.\d+)|(\*\*|[-+*/^()%]))")
_EXPR_CANDIDATE = re.compile(r"[\d.\s+\-*/^()%]+")
_MAX_EXPR_LEN = 256
_MAX_DEPTH = 32
_MAX_EXPONENT = 1000


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    expr = expr.rstrip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if m is None:
            raise ExpressionError(f"unexpected character at position {pos}: {expr[pos]!r}")
        tokens.append(m.group(1) or m.group(2))
        pos = m.end()
    return tokens


class _Parser:
    """Recursive descent over:  expr := term (('+'|'-') term)*
    term := unary (('*'|'/'|'%') unary)*   unary := ('-'|'+') unary | power
    power := atom (('^'|'**') unary)?       atom := NUMBER | '(' expr ')'
    """

    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens = tokens
        self.i = 0
        self.depth = 0

    def parse(self) -> Decimal:
        value = self._expr()
        if self.i != len(self.tokens):
            raise ExpressionError(f"unexpected token {self.tokens[self.i]!r}")
        return value

    def _peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _take(self) -> str:
        tok = self._peek()
        if tok is None:
            raise ExpressionError("unexpected end of expression")
        self.i += 1
        return tok

    def _expr(self) -> Decimal:
        value = self._term()
        while self._peek() in ("+", "-"):
            op = self._take()
            rhs = self._term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def _term(self) -> Decimal:
        value = self._unary()
        while self._peek() in ("*", "/", "%"):
            op = self._take()
            rhs = self._unary()
            if op == "*":
                value *= rhs
            elif op == "/":
                value /= rhs
            else:
                value %= rhs
        return value

    def _unary(self) -> Decimal:
        if self._peek() in ("-", "+"):
            op = self._take()
            self._enter()
            inner = self._unary()
            self.depth -= 1
            return -inner if op == "-" else inner
        return self._power()

    def _power(self) -> Decimal:
        base = self._atom()
        if self._peek() in ("^", "**"):
            self._take()
            exponent = self._unary()
            if abs(exponent) > _MAX_EXPONENT:
                raise ExpressionError("exponent too large")
            return base**exponent
        return base

    def _atom(self) -> Decimal:
        tok = self._take()
        if tok == "(":
            self._enter()
            value = self._expr()
            self.depth -= 1
            if self._take() != ")":
                raise ExpressionError("expected ')'")
            return value
        try:
            return Decimal(tok)
        except InvalidOperation as exc:
            raise ExpressionError(f"unexpected token {tok!r}") from exc

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise ExpressionError("expression nested too deeply")


def evaluate_arithmetic(expr: str) -> Decimal:
    if len(expr) > _MAX_EXPR_LEN:
        raise ExpressionError("expression too long")
    tokens = _tokenize(expr)
    if not tokens:
        raise ExpressionError("empty expression")
    with localcontext() as ctx:
        ctx.prec = 34
        ctx.traps[DivisionByZero] = True
        try:
            return _Parser(tokens).parse()
        except (DivisionByZero, InvalidOperation, Overflow) as exc:
            raise ExpressionError("undefined or out-of-range result") from exc


def _format_decimal(d: Decimal) -> str:
    if d == d.to_integral_value():
        return str(d.quantize(Decimal(1)))
    return format(d.normalize(), "f")


class Arithmetic:
    name = "arithmetic"

    def try_run(self, text: str) -> dict[str, JsonValue] | None:
        candidates = [
            c.strip()
            for c in _EXPR_CANDIDATE.findall(text)
            if re.search(r"\d", c) and re.search(r"\d\s*(\*\*|[-+*/^%])\s*[\d(.]", c)
        ]
        if not candidates:
            return None
        expr = max(candidates, key=len)
        try:
            value = evaluate_arithmetic(expr)
        except ExpressionError as exc:
            return {"expression": expr, "ok": False, "error": str(exc)}
        return {"expression": expr, "ok": True, "result": _format_decimal(value)}


# ---------------------------------------------------------------- unit conversion

_LINEAR_UNITS: dict[str, tuple[str, Decimal]] = {
    # unit -> (dimension, factor to SI base)
    "mm": ("length", Decimal("0.001")),
    "cm": ("length", Decimal("0.01")),
    "m": ("length", Decimal(1)),
    "km": ("length", Decimal(1000)),
    "in": ("length", Decimal("0.0254")),
    "ft": ("length", Decimal("0.3048")),
    "yd": ("length", Decimal("0.9144")),
    "mi": ("length", Decimal("1609.344")),
    "mg": ("mass", Decimal("0.000001")),
    "g": ("mass", Decimal("0.001")),
    "kg": ("mass", Decimal(1)),
    "lb": ("mass", Decimal("0.45359237")),
    "oz": ("mass", Decimal("0.028349523125")),
    "b": ("data", Decimal(1)),
    "kb": ("data", Decimal(1000)),
    "mb": ("data", Decimal(1000) ** 2),
    "gb": ("data", Decimal(1000) ** 3),
    "tb": ("data", Decimal(1000) ** 4),
    "kib": ("data", Decimal(1024)),
    "mib": ("data", Decimal(1024) ** 2),
    "gib": ("data", Decimal(1024) ** 3),
    "tib": ("data", Decimal(1024) ** 4),
}

_ALIASES: dict[str, str] = {
    "millimeter": "mm",
    "millimeters": "mm",
    "centimeter": "cm",
    "centimeters": "cm",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "inch": "in",
    "inches": "in",
    "foot": "ft",
    "feet": "ft",
    "yard": "yd",
    "yards": "yd",
    "mile": "mi",
    "miles": "mi",
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "kgs": "kg",
    "pound": "lb",
    "pounds": "lb",
    "lbs": "lb",
    "ounce": "oz",
    "ounces": "oz",
    "milligram": "mg",
    "milligrams": "mg",
    "byte": "b",
    "bytes": "b",
    "celsius": "c",
    "°c": "c",
    "fahrenheit": "f",
    "°f": "f",
    "kelvin": "k",
}

_TEMPS = {"c", "f", "k"}

_CONVERT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(°?[A-Za-z]+)\s+(?:to|in|into|as)\s+(°?[A-Za-z]+)\b", re.I)


def _norm_unit(u: str) -> str:
    u = u.lower()
    return _ALIASES.get(u, u)


def _to_kelvin(v: Decimal, unit: str) -> Decimal:
    if unit == "c":
        return v + Decimal("273.15")
    if unit == "f":
        return (v - 32) * 5 / 9 + Decimal("273.15")
    return v


def _from_kelvin(v: Decimal, unit: str) -> Decimal:
    if unit == "c":
        return v - Decimal("273.15")
    if unit == "f":
        return (v - Decimal("273.15")) * 9 / 5 + 32
    return v


class UnitConvert:
    name = "unit_convert"

    def try_run(self, text: str) -> dict[str, JsonValue] | None:
        m = _CONVERT.search(text)
        if m is None:
            return None
        value = Decimal(m.group(1))
        src, dst = _norm_unit(m.group(2)), _norm_unit(m.group(3))
        if src in _TEMPS and dst in _TEMPS:
            result = _from_kelvin(_to_kelvin(value, src), dst)
        elif src in _LINEAR_UNITS and dst in _LINEAR_UNITS:
            (dim_a, fa), (dim_b, fb) = _LINEAR_UNITS[src], _LINEAR_UNITS[dst]
            if dim_a != dim_b:
                return {"ok": False, "error": f"cannot convert {dim_a} to {dim_b}"}
            result = value * fa / fb
        else:
            return None
        return {
            "ok": True,
            "from": {"value": str(value), "unit": src},
            "to": {"value": format(round(result, 6).normalize(), "f"), "unit": dst},
        }


# ---------------------------------------------------------------- JSON validation

_JSON_INTENT = re.compile(r"\b(valid(ate)?|parse|lint|check)\b.*\bjson\b|\bjson\b.*\bvalid", re.I | re.S)


class JsonValidate:
    name = "json_validate"

    def try_run(self, text: str) -> dict[str, JsonValue] | None:
        if not _JSON_INTENT.search(text):
            return None
        starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
        if not starts:
            return None
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        snippet = text[start : end + 1] if end > start else text[start:]
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError as exc:
            return {
                "ok": True,
                "valid": False,
                "error": exc.msg,
                "line": exc.lineno,
                "column": exc.colno,
            }
        kind = type(parsed).__name__
        size = len(parsed) if isinstance(parsed, (dict, list)) else None
        return {"ok": True, "valid": True, "top_level_type": kind, "size": size}


# ---------------------------------------------------------------- date arithmetic

_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_INTENT = re.compile(r"\b(days?|between|until|since|difference)\b", re.I)


class DateDiff:
    name = "date_diff"

    def try_run(self, text: str) -> dict[str, JsonValue] | None:
        dates = _ISO_DATE.findall(text)
        if len(dates) != 2 or not _DATE_INTENT.search(text):
            return None
        try:
            a, b = date.fromisoformat(dates[0]), date.fromisoformat(dates[1])
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "from": dates[0], "to": dates[1], "days": (b - a).days}


# ---------------------------------------------------------------- registry


class DeterministicRegistry:
    """Tries handlers in order; the first that recognises the input wins."""

    def __init__(self, handlers: Sequence[DeterministicHandler]) -> None:
        self.handlers = list(handlers)

    def run(self, text: str) -> tuple[str, dict[str, JsonValue]] | None:
        for h in self.handlers:
            out = h.try_run(text)
            if out is not None:
                return h.name, out
        return None

    def can_handle(self, text: str) -> bool:
        return self.run(text) is not None


def default_registry() -> DeterministicRegistry:
    # Order matters: JSON and dates contain digits and '-' that arithmetic would claim.
    return DeterministicRegistry([JsonValidate(), DateDiff(), UnitConvert(), Arithmetic()])
