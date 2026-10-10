"""Commentary figure check: every figure in the model's commentary must come from the tool envelopes.

Pure functions (commentary and envelopes in, cleaned commentary out), applied by src/agent.py after
`drop_restated_headlines`. A sentence is removed when it has
- a number that appears nowhere in the envelopes of the conversation (formatting tolerated: 81.6 %,
  81.6%, 2,714,358, 2.7 million; a value matches when it rounds to the precision written, so 81.58
  supports 81.6 and 82; values below 1 also count as percentages, 0.385 = 38.5);
- in a compare_airports answer, a value credited to one airport that belongs to the other (the
  number is in the other airport's column and not in this one's);
- an airport name in parentheses next to a code (or the reverse) that is another airport's name in
  the envelopes, such as "Bradley (BTV)" when Bradley is BDL;
- a known misreading of a definition (MISREADINGS).
Ignored: years, periods (2025-01), numbers glued to letters (T-100, A320, 2nd) and whole numbers up
to IGNORE_SMALL_INTEGERS without a % sign (counts such as "3 airports", "12 months"). If nothing is left, the
commentary is replaced by WITHHELD. The data card stays the authoritative answer either way.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

WITHHELD = "The commentary was withheld because it contained figures that are not in the data."
# Whole numbers this small (without %) are counts, ordinals or months in prose, not figures from the card.
IGNORE_SMALL_INTEGERS = 12
YEARS = range(1900, 2101)
SCALES = {"million": Decimal(10) ** 6, "m": Decimal(10) ** 6, "billion": Decimal(10) ** 9, "bn": Decimal(10) ** 9,
          "thousand": Decimal(10) ** 3, "k": Decimal(10) ** 3}
# Definitions models have misread, with the reason shown in the check result.
MISREADINGS = (
    (re.compile(r"\d\s*%\s+of\s+(?:all\s+|the\s+)?long[- ]haul\s+(?:flights|departures)", re.IGNORECASE),
     "a long-haul share is the share of the airport's departures that are long haul, not a share of long-haul flights"),
)
# Words that do not identify an airport in a name ("Burlington International" = "Burlington").
GENERIC_NAME_WORDS = {"airport", "international", "intl", "regional", "municipal", "field", "the", "of", "county",
                      "jetport", "memorial", "air", "terminal"}

_DATE = re.compile(r"\b\d{4}[-‐‑‒–—/.]\d{1,2}(?:[-./]\d{1,2})?\b")
_NUMBER = re.compile(
    r"(?<![\w.,])(?<![A-Za-z]-)(\d{1,3}(?:,\d{3})+|\d+)(?:([.,])(\d+))?(?![\w])"
    r"(?:\s?(%|percent\b|pts\b|points\b|million\b|billion\b|bn\b|thousand\b|[mk]\b))?", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CODE = re.compile(r"\b[A-Z]{3}\b")
_NAME_BEFORE_CODE = re.compile(r"((?:[A-Z][\w.'-]*\s+){0,4}[A-Z][\w.'-]*)\s*\(([A-Z]{3})\)")
_CODE_BEFORE_NAME = re.compile(r"\b([A-Z]{3})\s*\(([^)]{3,60})\)")


@dataclass
class FigureCheck:
    text: str                                             # commentary, unsupported sentences removed
    removed: list[dict] = field(default_factory=list)     # {"sentence": ..., "reasons": [...]}
    withheld: bool = False                                # nothing was left: text is WITHHELD

    def as_dict(self) -> dict:
        return {"removed": self.removed, "withheld": self.withheld}


# --- numbers -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Written:
    """A number as written in the commentary: its value, decimals shown and scale word."""
    value: Decimal
    decimals: int
    scale: Decimal
    percent: bool
    start: int


def _written_numbers(text: str) -> list[Written]:
    """Numbers in `text` worth checking (see the module docstring for what is ignored)."""
    blanked = _DATE.sub(lambda m: " " * len(m.group()), text)
    found = []
    for m in _NUMBER.finditer(blanked):
        whole, frac, unit = m.group(1), m.group(3), (m.group(4) or "").lower()
        decimals = len(frac) if frac else 0
        try:
            value = Decimal(whole.replace(",", "") + (f".{frac}" if frac else ""))
        except InvalidOperation:
            continue
        percent = unit in ("%", "percent", "pts", "points")
        if not frac and not percent and not unit and (value <= IGNORE_SMALL_INTEGERS or int(value) in YEARS):
            continue
        found.append(Written(value, decimals, SCALES.get(unit, Decimal(1)), percent, m.start()))
    return found


def _numbers_in(value: Any, out: set[Decimal]) -> set[Decimal]:
    """Every number in an envelope (nested dicts and lists, numbers and numbers inside strings), as
    absolute values; values below 1 also as percentages."""
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, (int, float)):
        if value == value:
            number = abs(Decimal(str(value)))
            out.add(number)
            if number < 1:
                out.add(number * 100)
    elif isinstance(value, str):
        for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", value):
            try:
                _numbers_in(float(m.group().replace(",", "")), out)
            except ValueError:
                pass
    elif isinstance(value, dict):
        for v in value.values():
            _numbers_in(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _numbers_in(v, out)
    return out


def _supported(written: Written, allowed: set[Decimal]) -> bool:
    """True if some envelope value, in the written scale and rounded to the written decimals, equals it."""
    quantum = Decimal(1).scaleb(-written.decimals)
    for value in allowed:
        scaled = value / written.scale
        if scaled.quantize(quantum, rounding=ROUND_HALF_UP) == written.value or scaled == written.value:
            return True
    return False


# --- airports ------------------------------------------------------------------------------

def _airport_names(envelopes: list[Any]) -> dict[str, str]:
    """IATA -> name for every airport the envelopes name."""
    names: dict[str, str] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("iata"), str) and isinstance(value.get("name"), str):
                names.setdefault(value["iata"], value["name"])
            if isinstance(value.get("airport"), str) and isinstance(value.get("name"), str):
                names.setdefault(value["airport"], value["name"])
            if isinstance(value.get("names"), dict):
                for code, name in value["names"].items():
                    names.setdefault(code, name)
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    for env in envelopes:
        walk(env)
    return names


def _words(name: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", name.lower()) if w not in GENERIC_NAME_WORDS}


def _name_mismatches(sentence: str, names: dict[str, str]) -> list[str]:
    """'Bradley (BTV)' or 'BTV (Bradley International)': the name belongs to another airport in the
    envelopes and not to the code it is attached to. Names not in the envelopes are not judged."""
    pairs = [(m.group(1), m.group(2)) for m in _NAME_BEFORE_CODE.finditer(sentence)]
    pairs += [(m.group(2), m.group(1)) for m in _CODE_BEFORE_NAME.finditer(sentence)]
    reasons = []
    for written, code in pairs:
        if code not in names:
            continue
        # Words of the written name that are not in this airport's name ("Notably Bradley" -> notably,
        # bradley); flagged only if one of them is in another airport's name in the envelopes.
        foreign = _words(written) - _words(names[code])
        other = next((o for o, name in names.items() if o != code and foreign & _words(name)), None)
        if other:
            reasons.append(f"'{written.strip()}' is {other} ({names[other]}), not {code} ({names[code]})")
    return reasons


def _compare_columns(envelopes: list[Any]) -> list[tuple[dict[str, set[Decimal]], dict[str, list[str]]]]:
    """For each compare_airports envelope: numbers per airport (from facts with an `airport` key) and
    the words that mention each airport (its code and its name)."""
    out = []
    for env in envelopes:
        if not isinstance(env, dict):
            continue
        per_airport: dict[str, set[Decimal]] = {}
        for f in env.get("facts") or []:
            if isinstance(f, dict) and f.get("airport"):
                _numbers_in(f.get("value"), per_airport.setdefault(f["airport"], set()))
        if len(per_airport) == 2:
            names = (env.get("data") or {}).get("names") or {}
            out.append((per_airport, {c: [c, *([names[c]] if names.get(c) else [])] for c in per_airport}))
    return out


def _owner(position: int, mentions: list[tuple[int, str]]) -> str | None:
    """Airport a number belongs to in the sentence: the nearest mention before it, else the first after."""
    before = [code for start, code in mentions if start < position]
    if before:
        return before[-1]
    after = [code for start, code in mentions if start > position]
    return after[0] if after else None


def _swapped(sentence: str, numbers: list[Written], columns) -> list[str]:
    reasons = []
    for per_airport, aliases in columns:
        mentions = sorted((m.start(), code) for code, words in aliases.items() for word in words
                          for m in re.finditer(rf"\b{re.escape(word)}\b", sentence))
        for n in numbers:
            owner = _owner(n.start, mentions)
            if owner is None:
                continue
            other = next(c for c in per_airport if c != owner)
            if _supported(n, per_airport[other]) and not _supported(n, per_airport[owner]):
                reasons.append(f"{n.value}{'%' if n.percent else ''} is {other}'s figure, credited to {owner}")
    return reasons


# --- the check -----------------------------------------------------------------------------

def _sentences(commentary: str) -> list[list[str]]:
    """Lines, each split into sentences (a decimal point is never followed by a space)."""
    return [[s for s in _SENTENCE_END.split(line) if s.strip()] for line in commentary.split("\n")]


def sentence_problems(sentence: str, envelopes: list[Any], allowed: set[Decimal] | None = None) -> list[str]:
    """Why a sentence is unsupported (empty list: it is fine)."""
    allowed = allowed if allowed is not None else _numbers_in(envelopes, set())
    numbers = _written_numbers(sentence)
    reasons = [f"{n.value}{'%' if n.percent else ''} is not in the data" for n in numbers if not _supported(n, allowed)]
    reasons += _swapped(sentence, numbers, _compare_columns(envelopes))
    reasons += _name_mismatches(sentence, _airport_names(envelopes))
    reasons += [reason for pattern, reason in MISREADINGS if pattern.search(sentence)]
    return reasons


def check_commentary(commentary: str, envelopes: list[Any]) -> FigureCheck:
    """Remove sentences with unsupported figures; WITHHELD if nothing is left (see module docstring).

    `envelopes` should hold every tool result of the conversation: a follow-up may quote an
    earlier card. Non-dict results are ignored."""
    envelopes = [e for e in envelopes if isinstance(e, dict)]
    allowed = _numbers_in(envelopes, set())
    removed, lines = [], []
    for sentences in _sentences(commentary):
        kept = []
        for sentence in sentences:
            reasons = sentence_problems(sentence, envelopes, allowed)
            if reasons:
                removed.append({"sentence": sentence.strip(), "reasons": reasons})
            else:
                kept.append(sentence)
        if kept or not sentences:
            lines.append(" ".join(kept))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if removed and not re.search(r"\w", text):
        return FigureCheck(WITHHELD, removed, True)
    return FigureCheck(text, removed, False)

