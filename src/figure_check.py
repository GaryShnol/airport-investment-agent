"""Commentary figure check: every figure in the model's commentary must come from the tool envelopes.

Pure functions (commentary and envelopes in, cleaned commentary out), applied by src/agent.py after
`drop_restated_headlines`. A sentence is removed when it has
- a number that appears nowhere in the envelopes of the conversation (formatting tolerated: 81.6 %,
  81.6%, 2,714,358, 2.7 million; a value matches when it rounds to the precision written, so 81.58
  supports 81.6 and 82; values below 1 also count as percentages, 0.385 = 38.5);
- in a compare_airports answer, a value plainly credited to one airport ("LAX has 13.2%", "13.2% at
  LAX") that belongs to the other (in the other airport's column and not in this one's); airports are
  recognized by code, name and the aliases the tools list (city served such as "Santa Ana", "LA");
- an airport name in parentheses next to a code (or the reverse) that is another airport's name in
  the envelopes, such as "Bradley (BTV)" when Bradley is BDL;
- a known misreading of a definition (MISREADINGS);
- a month count ("11 of 12 months") that no envelope states ("10 of 12"): small whole numbers are
  otherwise not checked as figures;
- a month or year outside the data periods of the envelopes, or a month range that is neither one of
  them nor made only of months the envelopes name ("Mar 2025 - Dec 2025" when the data is
  2025-01..2025-12; "January to December 2025", "2025-01..2025-12" and, with peak months 2025-06 and
  2025-07, "June-July" pass). Single months and years pass when a period covers them or the envelopes
  name them (peak months, the upstream month of a freshness note, Census vintages).
Ignored as figures: years, periods (2025-01), numbers glued to letters (T-100, A320, 2nd) and whole numbers up
to IGNORE_SMALL_INTEGERS without a % sign (counts such as "3 airports", "12 months"). The data card stays the
authoritative answer either way.

Restricted mode (`restrict_commentary`, for the tools in `commentary.restricted_tools`: rankings and
comparisons). Presence of a number cannot show that an ordering or a direction is right ("BDL ranks
first", "LAX has the shorter taxi-out" with real figures from the card). For these tools the card
and headline give the result, and the commentary may only add caveats or a next step: every
sentence with a digit, an airport code, name or alias (city served) from the envelopes is dropped, and so
are tables, headings and rules (a restated card).

After either check, `finish_commentary` (thresholds and wording from `commentary` in config/weights.yaml)
drops the fragments a removal leaves behind ("F.", "The ranking is as follows:") and, when nothing is
left, shows one fixed fallback line; when the model wrote nothing a restricted answer shows no
commentary block. `removal_note` is the one-line caption: the count and each distinct reason once.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

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
# A sentence ends at . ! ? and a space, but not after an initial ("T. F. Green" is one name).
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!\b[A-Z]\.)\s+")
_NAME_BEFORE_CODE = re.compile(r"((?:[A-Z][\w.'-]*\s+){0,4}[A-Z][\w.'-]*)\s*\(([A-Z]{3})\)")
_CODE_BEFORE_NAME = re.compile(r"\b([A-Z]{3})\s*\(([^)]{3,60})\)")


# Restricted mode: markdown structure that only restates a card (table rows, headings, rules, bold-only lines).
_STRUCTURE_LINE = re.compile(r"^\s*(?:\||#|[-*_]{3,}\s*$|\*\*[^*]+\*\*:?\s*$)")
_DIGIT = re.compile(r"\d")


@dataclass
class FigureCheck:
    text: str                                             # commentary, unsupported sentences removed
    removed: list[dict] = field(default_factory=list)     # {"sentence": ..., "reasons": [...]}
    withheld: bool = False                                # removals left nothing: text is the fallback line
    mode: str = "figures"                                 # "figures" or "restricted"
    hidden: bool = False                                  # restricted, and the model wrote nothing: no block

    def as_dict(self) -> dict:
        return {"mode": self.mode, "removed": self.removed, "withheld": self.withheld, "hidden": self.hidden}


# --- numbers -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Written:
    """A number as written in the commentary: its value, decimals shown and scale word."""
    value: Decimal
    decimals: int
    scale: Decimal
    percent: bool
    start: int
    end: int


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
        found.append(Written(value, decimals, SCALES.get(unit, Decimal(1)), percent, m.start(), m.end()))
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


def _airport_aliases(envelopes: list[Any]) -> dict[str, list[str]]:
    """IATA -> other words naming it in prose (city served, "LA", the user's words), from the `aliases`
    map that compare_airports and rank_airports put in their data."""
    aliases: dict[str, list[str]] = {}
    for env in envelopes:
        data = env.get("data") if isinstance(env, dict) else None
        found = data.get("aliases") if isinstance(data, dict) else None
        if isinstance(found, dict):
            for code, terms in found.items():
                aliases.setdefault(code, []).extend(t for t in terms or [] if isinstance(t, str) and t.strip())
    return aliases


def _alias_pattern(term: str) -> re.Pattern:
    """A whole-word pattern for an alias: case-insensitive, except short ones ("LA", not "la")."""
    return re.compile(rf"\b{re.escape(term.strip())}\b", 0 if len(term.strip()) <= 3 else re.IGNORECASE)


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
            aliases = _airport_aliases([env])
            out.append((per_airport, {c: [c, *([names[c]] if names.get(c) else []), *aliases.get(c, [])]
                                      for c in per_airport}))
    return out


# Text between an airport and a number that ends the attribution: a new clause or a comparison
# ("LAX has 81.6% vs 79.7%": the second value is the other airport's).
_ATTRIBUTION_BREAK = re.compile(r"[,;]|\b(?:and|while|whereas|but|vs|versus|than|compared|against|unlike|or)\b",
                                re.IGNORECASE)
# "18.1% at LAX", "17.7 min for LAX": a number followed by its airport.
_FOLLOWED_BY = r"^\s*(?:%|percent|pts|points|min(?:utes)?)?\s*(?:at|for|in|from)\s+"


def _owner(sentence: str, n: Written, mentions: list[tuple[int, int, str]]) -> str | None:
    """Airport a number is credited to, only when the sentence says so plainly: the number is
    followed by "at/for <airport>", or it comes after an airport with no clause break or comparison
    word in between ("LAX has 18.1%", "LAX: 18.1%"). Anything else ("its 19.2% versus 18.1% at
    LAX") is not judged: removing a correct sentence is worse than missing a swap."""
    after = sentence[n.end:]
    for start, _end, code in mentions:
        if start >= n.end and re.match(_FOLLOWED_BY + r"$", after[:start - n.end]):
            return code
    before = [(end, code) for _start, end, code in mentions if end <= n.start]
    if before:
        end, code = before[-1]
        if not _ATTRIBUTION_BREAK.search(sentence[end:n.start]):
            return code
    return None


def _swapped(sentence: str, numbers: list[Written], columns) -> list[str]:
    reasons = []
    for per_airport, aliases in columns:
        mentions = sorted((m.start(), m.end(), code) for code, words in aliases.items() for word in words
                          for m in re.finditer(rf"\b{re.escape(word)}\b", sentence))
        for n in numbers:
            owner = _owner(sentence, n, mentions)
            if owner is None:
                continue
            other = next(c for c in per_airport if c != owner)
            if _supported(n, per_airport[other]) and not _supported(n, per_airport[owner]):
                reasons.append(f"{n.value}{'%' if n.percent else ''} is {other}'s figure, credited to {owner}")
    return reasons


# --- periods -------------------------------------------------------------------------------

_MONTH_NAMES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
                "oct": 10, "nov": 11, "dec": 12}
# Capitalized month names and abbreviations ("Mar", "March", "Sept."); lower-case "may" is a verb.
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
          r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
_RANGE_SEP = r"\s*(?:\.\.|[-‐‑‒–—]|\bto\b|\bthrough\b|\bthru\b|\buntil\b)\s*"
_ISO_RANGE = re.compile(rf"\b(\d{{4}})-(\d{{2}}){_RANGE_SEP}(\d{{4}})-(\d{{2}})\b")
_NAMED_RANGE = re.compile(rf"\b({_MONTH})(?:\s+(\d{{4}}))?{_RANGE_SEP}({_MONTH})(?:\s+(\d{{4}}))?(?![\w])")
_NAMED_MONTH = re.compile(rf"\b({_MONTH})(?:\s+(\d{{4}}))?(?![\w])")
_ISO_MONTH = re.compile(r"\b(\d{4})-(\d{2})\b(?!-\d)")
_YEAR = re.compile(r"(?<![\d,.-])\b((?:19|20)\d{2})\b(?![\d,.]\d|-\d)")
_PERIOD_IN_DATA = re.compile(r"(\d{4})-(\d{2})\.\.(\d{4})-(\d{2})")


def _ym(year: int | str, month: int) -> int:
    return int(year) * 12 + month - 1


def _month(name: str) -> int:
    return _MONTH_NAMES[name.lower()[:3]]


@dataclass(frozen=True)
class Periods:
    """What the envelopes cover: data periods (as month indexes), months and years they name."""
    ranges: frozenset[tuple[int, int]]
    months: frozenset[int]
    years: frozenset[int]

    def covers(self, ym: int) -> bool:
        return ym in self.months or any(a <= ym <= b for a, b in self.ranges)

    def covers_year(self, year: int) -> bool:
        return year in self.years or any(a // 12 <= year <= b // 12 for a, b in self.ranges)

    def names_all(self, span: tuple[int, int]) -> bool:
        """Every month of the span is named in the envelopes (peak months: "June-July 2025")."""
        return span[0] <= span[1] and all(ym in self.months for ym in range(span[0], span[1] + 1))

    def names_all_months(self, first: int, last: int) -> bool:
        """Every month of the year from `first` to `last` (wrapping past December) is named in the
        envelopes in some year inside a data period ("June-July" with peak months 2025-06, 2025-07)."""
        named = {ym % 12 + 1 for ym in self.months if any(a <= ym <= b for a, b in self.ranges)}
        months = range(first, last + 1) if first <= last else [*range(first, 13), *range(1, last + 1)]
        return all(m in named for m in months)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def data_periods(envelopes: list[Any]) -> Periods:
    text = "\n".join(_strings(envelopes))
    ranges = {(_ym(a, int(b)), _ym(c, int(d))) for a, b, c, d in _PERIOD_IN_DATA.findall(text)}
    months = {_ym(y, int(m)) for y, m in re.findall(r"\b(\d{4})-(\d{2})\b", text) if 1 <= int(m) <= 12}
    years = {int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", text)}
    return Periods(frozenset(ranges), frozenset(months), frozenset(years))


def _blank(text: str, m: re.Match) -> str:
    return text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]


def period_problems(sentence: str, periods: Periods) -> list[str]:
    """Months, month ranges and years the data does not cover (empty list: fine, or no period known)."""
    if not periods.ranges:
        return []
    reasons, text = [], sentence
    for m in list(_ISO_RANGE.finditer(text)):
        span = (_ym(m.group(1), int(m.group(2))), _ym(m.group(3), int(m.group(4))))
        if span not in periods.ranges:
            reasons.append(f"'{m.group().strip().rstrip('.')}' is not the data period")
        text = _blank(text, m)
    for m in list(_NAMED_RANGE.finditer(text)):
        (m1, y1), (m2, y2) = (_month(m.group(1)), m.group(2)), (_month(m.group(3)), m.group(4))
        if y1 or y2:
            y2 = int(y2 or y1)
            y1 = int(y1) if y1 else (y2 if m1 <= m2 else y2 - 1)
            span = (_ym(y1, m1), _ym(y2, m2))
            ok = span in periods.ranges or periods.names_all(span)
        else:  # "March to December": a period's first and last month, or months the envelopes all name
            ok = (any(a % 12 + 1 == m1 and b % 12 + 1 == m2 for a, b in periods.ranges)
                  or periods.names_all_months(m1, m2))
        if not ok:
            reasons.append(f"'{m.group().strip().rstrip('.')}' is not the data period")
        text = _blank(text, m)
    for m in list(_NAMED_MONTH.finditer(text)):
        month, year = _month(m.group(1)), m.group(2)
        if year:
            if not periods.covers(_ym(year, month)):
                reasons.append(f"'{m.group().strip().rstrip('.')}' is outside the data period")
            text = _blank(text, m)
        elif m.group(1) != "May" and not any(month - 1 in {ym % 12 for ym in range(a, b + 1)}
                                             for a, b in periods.ranges):
            reasons.append(f"'{m.group().strip().rstrip('.')}' is outside the data period")
    for m in list(_ISO_MONTH.finditer(text)):
        if 1 <= int(m.group(2)) <= 12 and not periods.covers(_ym(m.group(1), int(m.group(2)))):
            reasons.append(f"'{m.group()}' is outside the data period")
        text = _blank(text, m)
    reasons += [f"{y} is outside the data period" for y in dict.fromkeys(_YEAR.findall(text))
                if not periods.covers_year(int(y))]
    return reasons


# --- month counts --------------------------------------------------------------------------

# "11 of 12 months" in the commentary; whole numbers up to IGNORE_SMALL_INTEGERS are not checked as
# figures, so a wrong count of months (11 when the card says 10 of 12) needs its own check.
_MONTH_COUNT = re.compile(r"\b(\d{1,2})\s+(?:out\s+)?of\s+(?:the\s+)?(\d{1,2})\s+months?\b", re.IGNORECASE)
# "10 of 12" as the envelopes write counts ("Months with load factor above 80%: 10 of 12").
_COUNT_IN_DATA = re.compile(r"\b(\d+)\s+(?:out\s+)?of\s+(?:the\s+)?(\d+)\b")


def data_counts(envelopes: list[Any]) -> frozenset[tuple[int, int]]:
    """Every "N of M" pair the envelopes state."""
    return frozenset((int(a), int(b)) for a, b in _COUNT_IN_DATA.findall("\n".join(_strings(envelopes))))


def count_problems(sentence: str, counts: frozenset[tuple[int, int]]) -> list[str]:
    """Month counts ("N of M months") that no envelope states."""
    return [f"'{m.group()}' is not in the data" for m in _MONTH_COUNT.finditer(sentence)
            if (int(m.group(1)), int(m.group(2))) not in counts]


# --- the check -----------------------------------------------------------------------------

def _sentences(commentary: str) -> list[list[str]]:
    """Lines, each split into sentences (a decimal point is never followed by a space)."""
    return [[s for s in _SENTENCE_END.split(line) if s.strip()] for line in commentary.split("\n")]


def sentence_problems(sentence: str, envelopes: list[Any], allowed: set[Decimal] | None = None,
                      periods: Periods | None = None, counts: frozenset[tuple[int, int]] | None = None) -> list[str]:
    """Why a sentence is unsupported (empty list: it is fine)."""
    allowed = allowed if allowed is not None else _numbers_in(envelopes, set())
    periods = periods if periods is not None else data_periods(envelopes)
    counts = counts if counts is not None else data_counts(envelopes)
    numbers = _written_numbers(sentence)
    reasons = [f"{n.value}{'%' if n.percent else ''} is not in the data" for n in numbers if not _supported(n, allowed)]
    reasons += _swapped(sentence, numbers, _compare_columns(envelopes))
    reasons += _name_mismatches(sentence, _airport_names(envelopes))
    reasons += [reason for pattern, reason in MISREADINGS if pattern.search(sentence)]
    reasons += period_problems(sentence, periods)
    reasons += count_problems(sentence, counts)
    return reasons


def _filter(commentary: str, judge) -> tuple[str, list[dict]]:
    """Commentary without the sentences `judge` (sentence -> reasons) objects to, and what was removed."""
    removed, lines = [], []
    for sentences in _sentences(commentary):
        kept = []
        for sentence in sentences:
            reasons = judge(sentence)
            if reasons:
                removed.append({"sentence": sentence.strip(), "reasons": reasons})
            else:
                kept.append(sentence)
        if kept or not sentences:
            lines.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), removed


def check_commentary(commentary: str, envelopes: list[Any]) -> FigureCheck:
    """Remove sentences with unsupported figures (see module docstring); then `finish_commentary`.

    `envelopes` should hold every tool result of the conversation: a follow-up may quote an
    earlier card. Non-dict results are ignored."""
    envelopes = [e for e in envelopes if isinstance(e, dict)]
    allowed, periods, counts = _numbers_in(envelopes, set()), data_periods(envelopes), data_counts(envelopes)
    text, removed = _filter(commentary, lambda s: sentence_problems(s, envelopes, allowed, periods, counts))
    return FigureCheck(text, removed)


def _airport_mentions(names: dict[str, str],
                      aliases: dict[str, list[str]] | None = None) -> list[tuple[str, re.Pattern]]:
    """(label, pattern) for every airport in the envelopes: its code, its full name, each distinctive
    word of the name as written ("Bradley", "Burlington"; generic and short words skipped) and its
    aliases (city served such as "Santa Ana", "LA", the user's words; see `_airport_aliases`)."""
    out = []
    for code, name in names.items():
        words = [w for w in re.findall(r"[A-Z][\w'-]+", name) if len(w) >= 4 and w.lower() not in GENERIC_NAME_WORDS]
        for term in dict.fromkeys([code, name, *words]):
            out.append((code, re.compile(rf"\b{re.escape(term)}\b")))
    for code, terms in (aliases or {}).items():
        out += [(code, _alias_pattern(term)) for term in dict.fromkeys(terms)]
    return out


def restriction_problems(sentence: str, mentions: list[tuple[str, re.Pattern]]) -> list[str]:
    """Why a sentence is not allowed in restricted mode (empty list: a caveat or next step, kept)."""
    if _STRUCTURE_LINE.match(sentence):
        return ["table, heading or rule (restates the card)"]
    reasons = ["contains a figure"] if _DIGIT.search(sentence) else []
    named = sorted({code for code, pattern in mentions if pattern.search(sentence)})
    return reasons + ([f"names an airport ({', '.join(named)})"] if named else [])


def restrict_commentary(commentary: str, envelopes: list[Any]) -> FigureCheck:
    """Restricted mode (see module docstring): keep only sentences with no digit, no airport code and
    no airport name from the envelopes; then `finish_commentary`."""
    envelopes = [e for e in envelopes if isinstance(e, dict)]
    mentions = _airport_mentions(_airport_names(envelopes), _airport_aliases(envelopes))
    text, removed = _filter(commentary, lambda s: restriction_problems(s, mentions))
    return FigureCheck(text, removed, mode="restricted")


# --- after the check -----------------------------------------------------------------------

FRAGMENT = "fragment left after the removals"
# A sentence end, possibly followed by closing quotes, brackets or markdown emphasis.
_TERMINAL = re.compile(r"[.!?…][\"'”’)\]*_]*$")


def fragment_problems(sentence: str, min_words: int) -> list[str]:
    """Why a leftover is not a sentence: fewer than `min_words` words, ends with ":", or no sentence end."""
    text = sentence.strip()
    if len(re.findall(r"\w+", text)) < min_words or text.endswith(":") or not _TERMINAL.search(text):
        return [FRAGMENT]
    return []


def finish_commentary(check: FigureCheck, min_words: int, fallback: str) -> FigureCheck:
    """Clean up after a check: if it removed anything, drop the fragments left behind; if removals
    left nothing, the `fallback` line (`withheld`). Nothing written at all: a restricted answer is
    `hidden` (the card is the whole answer), a figures answer stays empty for the agent's fallback."""
    text, removed = check.text, check.removed
    if removed:
        text, fragments = _filter(text, lambda s: fragment_problems(s, min_words))
        removed = [*removed, *fragments]
    if re.search(r"\w", text):
        return replace(check, text=text, removed=removed)
    if removed:
        return replace(check, text=fallback, removed=removed, withheld=True)
    return replace(check, text="", hidden=check.mode == "restricted")


def removal_note(removed: list[dict], template: str) -> str:
    """One caption for the removed sentences (empty when none): the count and each distinct reason
    once, without the per-sentence detail in parentheses ("names an airport (BOS, PVD)" ->
    "names an airport"). The sentences and full reasons stay in the check result and the log."""
    if not removed:
        return ""
    reasons = dict.fromkeys(re.sub(r"\s*\([^)]*\)$", "", r) for item in removed for r in item["reasons"])
    return template.format(count=len(removed), reasons="; ".join(reasons))

