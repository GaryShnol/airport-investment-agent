"""Markdown rendering of tool envelopes: the figures a user sees come from here, not from the model.

Pure functions (envelope in, markdown out), shared by the Streamlit app and the CLI. The model's
text is shown after this, labeled as commentary. Each part is its own function so the app can
place the excluded list and the assumptions in collapsible sections.
"""
from __future__ import annotations

import json
import re
from typing import Any

# rank_airports also lists one fact per ranked airport ("#1 BOS (...)"); the table shows those rows.
_RANKING_FACT = re.compile(r"^#\d+ ")


def _cell(value: Any) -> str:
    """Table cell text: pipes escaped, newlines flattened."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_facts(envelope: dict) -> str:
    """Facts card: one row per fact, then freshness unless a fact already states it.

    Required facts are marked with *; ranking rows are left to `render_ranking`."""
    rows = [[f["label"] + (" *" if f.get("required") else ""), f["value"]]
            for f in envelope.get("facts") or [] if not _RANKING_FACT.match(f["label"])]
    if envelope.get("freshness") and not any(f["label"] == "Freshness" for f in envelope.get("facts") or []):
        rows.append(["Freshness", envelope["freshness"]])
    return _table(["Fact", "Value"], rows) if rows else ""


def render_ranking(envelope: dict) -> str:
    """Ranking table from rank_airports `data.ranked` (raw values, not percentiles), plus ties."""
    data = envelope.get("data") or {}
    ranked = data.get("ranked") if isinstance(data, dict) else None
    if not ranked:
        return ""
    rows = []
    for r in ranked:
        comp = r.get("components", {})
        passengers = comp.get("scale", {}).get("passengers")
        rows.append([r["rank"], r["iata"], r["name"], f"{r['score']:.3f}",
                     "n/a" if passengers is None else f"{passengers:,}",
                     _pct(comp.get("utilization", {}).get("load_factor_pct")),
                     _pct(comp.get("growth", {}).get("growth_pct"), signed=True),
                     r["confidence"]])
    table = _table(["Rank", "Airport", "Name", "Score", "Passengers", "Load factor", "Growth", "Confidence"], rows)
    total = data.get("ranked_total", len(ranked))
    if total > len(ranked):
        table += f"\n\nTop {len(ranked)} of {total} ranked airports."
    for group in data.get("ties") or []:
        table += f"\n\nTie (equal score): {', '.join(group)}."
    return table


def render_excluded(envelope: dict) -> str:
    """Airports left out of a ranking, with the reason, as a short list."""
    data = envelope.get("data") or {}
    excluded = data.get("excluded") if isinstance(data, dict) else None
    if not excluded:
        return ""
    lines = [f"- {item}" for item in excluded]
    more = data.get("excluded_total", len(excluded)) - len(excluded)
    if more > 0:
        lines.append(f"- and {more} more, smaller still")
    return "\n".join(lines)


def excluded_title(envelope: dict) -> str:
    data = envelope.get("data") or {}
    return f"Excluded airports ({data.get('excluded_total', len(data.get('excluded') or []))})"


def render_assumptions(envelope: dict) -> str:
    return "\n".join(f"- {a}" for a in envelope.get("assumptions") or [])


def render_error(envelope: dict) -> str:
    text = f"**Tool error:** {envelope['error']}"
    if envelope.get("candidates"):
        text += f"\n\nCandidates: {', '.join(map(str, envelope['candidates']))}"
    if envelope.get("default"):
        text += f" (default: {envelope['default']})"
    return text


def render_envelope(envelope: Any, title: str | None = None) -> str:
    """Whole envelope as markdown: title, error or facts card, ranking, excluded, assumptions."""
    parts = [f"**{title}**"] if title else []
    if not isinstance(envelope, dict):  # a tool result that was not JSON
        parts.append(f"```\n{envelope if isinstance(envelope, str) else json.dumps(envelope)}\n```")
        return "\n\n".join(parts)
    if envelope.get("error"):
        parts.append(render_error(envelope))
        return "\n\n".join(parts)
    parts += [render_facts(envelope), render_ranking(envelope)]
    if excluded := render_excluded(envelope):
        parts.append(f"{excluded_title(envelope)}:\n\n{excluded}")
    if assumptions := render_assumptions(envelope):
        parts.append(f"Assumptions:\n\n{assumptions}")
    return "\n\n".join(p for p in parts if p)


def render_reply(tool_calls: list[dict], envelopes: list[Any]) -> str:
    """All envelopes of one turn, each titled with its tool name; empty when no tool ran."""
    names = [c["name"] for c in tool_calls]
    return "\n\n---\n\n".join(render_envelope(env, names[i] if i < len(names) else None)
                              for i, env in enumerate(envelopes))
