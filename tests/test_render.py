"""Golden markdown for src/render.py: hand-written envelopes in, exact markdown out."""
from src.render import render_envelope, render_excluded, render_facts, render_ranking, render_reply

METRICS = {
    "facts": [
        {"label": "Airport", "value": "ANC (Ted Stevens Anchorage International Airport)", "required": False},
        {"label": "Long-haul share >= 2,500 mi, all flights (including cargo)", "value": "43.2%", "required": True},
        {"label": "Long-haul share >= 2,500 mi, passenger service", "value": "11.5%", "required": True},
        {"label": "Data period", "value": "2025-01..2025-12", "required": True},
        {"label": "Confidence", "value": "high", "required": True},
    ],
    "data": {"airport": "ANC"},
    "data_period": "2025-01..2025-12", "freshness": "unknown",
    "assumptions": ["long haul = segment distance >= 2500 mi, share of departures"],
    "confidence": "high",
}

RANKING = {
    "facts": [
        {"label": "Weights used (renormalized without congestion)", "value": "utilization 0.385, scale 0.231",
         "required": True},
        {"label": "#1 BOS (Boston Logan International Airport)", "value": "score 0.9", "required": False},
        {"label": "Data period", "value": "2025-01..2025-12", "required": True},
        {"label": "Confidence", "value": "medium", "required": True},
    ],
    "data": {
        "ranked": [
            {"rank": 1, "iata": "BOS", "name": "Boston Logan International Airport", "score": 0.9,
             "components": {"utilization": {"load_factor_pct": 84.1, "percentile": 80.0},
                            "scale": {"passengers": 20123456, "percentile": 99.0},
                            "growth": {"growth_pct": 2.5, "percentile": 40.0}},
             "confidence": "medium"},
            {"rank": 2, "iata": "PWM", "name": "Portland International Jetport", "score": 0.7123,
             "components": {"utilization": {"load_factor_pct": 82.4, "percentile": 70.0},
                            "growth": {"growth_pct": -1.0, "percentile": 10.0}},
             "confidence": "low"},
        ],
        "ranked_total": 6, "ties": [],
        "excluded": ["PSM: 85,102 passengers, below the 100,000 floor", "HYA: 5 passengers, below the 100,000 floor"],
        "excluded_total": 4,
    },
    "data_period": "2025-01..2025-12", "freshness": "current",
    "assumptions": ["score is a capacity-pressure proxy"], "confidence": "medium",
}


def test_facts_card_golden():
    assert render_facts(METRICS) == (
        "| Fact | Value |\n"
        "|---|---|\n"
        "| Airport | ANC (Ted Stevens Anchorage International Airport) |\n"
        "| Long-haul share >= 2,500 mi, all flights (including cargo) * | 43.2% |\n"
        "| Long-haul share >= 2,500 mi, passenger service * | 11.5% |\n"
        "| Data period * | 2025-01..2025-12 |\n"
        "| Confidence * | high |\n"
        "| Freshness | unknown |"
    )


def test_ranking_table_golden():
    assert render_ranking(RANKING) == (
        "| Rank | Airport | Name | Score | Passengers | Load factor | Growth | Confidence |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 1 | BOS | Boston Logan International Airport | 0.900 | 20,123,456 | 84.1% | +2.5% | medium |\n"
        "| 2 | PWM | Portland International Jetport | 0.712 | n/a | 82.4% | -1.0% | low |\n"
        "\n"
        "Top 2 of 6 ranked airports."
    )


def test_ranking_card_leaves_airport_rows_to_the_table():
    card = render_facts(RANKING)
    assert "#1 BOS" not in card and "| Weights used (renormalized without congestion) * |" in card


def test_excluded_list_counts_the_rest():
    assert render_excluded(RANKING) == (
        "- PSM: 85,102 passengers, below the 100,000 floor\n"
        "- HYA: 5 passengers, below the 100,000 floor\n"
        "- and 2 more, smaller still"
    )


def test_envelope_order_and_pipes_escaped():
    env = {**METRICS, "facts": [{"label": "Note", "value": "a | b", "required": False}]}
    md = render_envelope(env, "airport_metrics")
    assert md.startswith("**airport_metrics**\n\n| Fact | Value |")
    assert "| Note | a \\| b |" in md
    assert md.endswith("Assumptions:\n\n- long haul = segment distance >= 2500 mi, share of departures")


def test_error_envelope_and_non_json_result():
    err = {"data": None, "facts": [], "error": "'LA' is ambiguous", "candidates": ["LAX", "BUR"], "default": "LAX"}
    assert render_envelope(err) == "**Tool error:** 'LA' is ambiguous\n\nCandidates: LAX, BUR (default: LAX)"
    assert render_envelope("boom") == "```\nboom\n```"


def test_reply_titles_each_envelope_with_its_tool():
    md = render_reply([{"name": "resolve_airport", "args": {}}, {"name": "airport_metrics", "args": {}}],
                      [METRICS, METRICS])
    assert md.count("**resolve_airport**") == 1 and "\n\n---\n\n**airport_metrics**" in md
    assert render_reply([], []) == ""


def test_headline_is_rendered_above_the_card():
    md = render_envelope({**METRICS, "headline": "43.2% of all departures from ANC are long haul."}, "airport_metrics")
    assert md.startswith("**airport_metrics**\n\n> 43.2% of all departures from ANC are long haul.\n\n| Fact | Value |")
    err = {"data": None, "facts": [], "error": "Unknown region", "headline": "Unknown region"}
    assert render_envelope(err) == "**Tool error:** Unknown region"  # not shown twice


def test_scenario_ranking_shows_base_rank_column():
    env = {**RANKING, "data": {**RANKING["data"], "ranked": [{**r, "base_rank": 2 - i}
                                                             for i, r in enumerate(RANKING["data"]["ranked"])]}}
    lines = render_ranking(env).splitlines()
    assert lines[0] == "| Rank | Base rank | Airport | Name | Score | Passengers | Load factor | Growth | Confidence |"
    assert lines[2].startswith("| 1 | 2 | BOS |")


def test_ranking_shows_delayed_departures_when_congestion_is_scored():
    ranked = [{**r, "components": {**r["components"],
                                   **({"congestion": {"dep_delay_rate_pct": 22.8, "taxi_out_min": 21.2,
                                                      "nas_min_per_arrival": 6.96, "percentile": 95.0}}
                                      if r["iata"] == "BOS" else {})}}
              for r in RANKING["data"]["ranked"]]
    lines = render_ranking({**RANKING, "data": {**RANKING["data"], "ranked": ranked}}).splitlines()
    assert lines[0] == ("| Rank | Airport | Name | Score | Passengers | Load factor | Growth | Delayed departures "
                        "| Confidence |")
    assert lines[2].endswith("| +2.5% | 22.8% | medium |")
    assert lines[3].endswith("| -1.0% | n/a | low |")  # PWM without On-Time data


COMPARISON = {
    "headline": "LAX vs SNA: ...",
    "facts": [
        {"label": "LAX: Departures delayed >= 15 min", "value": "18.1% ± 0.17 pts", "required": True, "airport": "LAX"},
        {"label": "SNA: Departures delayed >= 15 min", "value": "19.2% ± 0.37 pts", "required": True, "airport": "SNA"},
        {"label": "Verdict: Departures delayed >= 15 min", "value": "SNA higher", "required": False,
         "metric": "dep_delay_rate"},
        {"label": "Data period", "value": "2025-01..2025-12", "required": True},
    ],
    "data": {"airports": ["LAX", "SNA"], "names": {"LAX": "Los Angeles Intl", "SNA": "John Wayne"},
             "metrics": [{"key": "dep_delay_rate", "label": "Departures delayed >= 15 min", "verdict": "SNA higher",
                          "cells": {"LAX": {"display": "18.1% ± 0.17 pts"}, "SNA": {"display": "19.2% ± 0.37 pts"}}},
                         {"key": "passengers", "label": "Passengers", "verdict": "LAX higher (full count)",
                          "cells": {"LAX": {"display": "36,778,508"}, "SNA": {"display": "5,532,646"}}}]},
    "data_period": "2025-01..2025-12", "freshness": "unknown", "assumptions": [], "confidence": "medium",
}


def test_comparison_is_a_two_column_table_and_its_facts_leave_the_card():
    out = render_envelope(COMPARISON)
    assert ("LAX = Los Angeles Intl; SNA = John Wayne\n\n"
            "| Metric | LAX | SNA | Verdict |\n|---|---|---|---|\n"
            "| Departures delayed >= 15 min | 18.1% ± 0.17 pts | 19.2% ± 0.37 pts | SNA higher |\n"
            "| Passengers | 36,778,508 | 5,532,646 | LAX higher (full count) |") in out
    assert render_facts(COMPARISON) == ("| Fact | Value |\n|---|---|\n| Data period * | 2025-01..2025-12 |\n"
                                        "| Freshness | unknown |")


def test_demand_gap_why_is_rendered_as_a_list_under_the_card():
    env = {"headline": "PROXY, not a measurement: ...", "facts": [], "data": {"why": ["First.", "Second."]},
           "assumptions": [], "confidence": "medium"}
    assert "Why (built only from the facts above):\n\n- First.\n- Second." in render_envelope(env)


def test_ranking_shows_stability_columns_when_computed():
    env = {**RANKING, "data": {**RANKING["data"], "ranked": [
        {**RANKING["data"]["ranked"][0], "stability": {"top_n": 3, "in_top_n_pct": 100.0, "min_rank": 1, "max_rank": 1}},
        {**RANKING["data"]["ranked"][1], "stability": {"top_n": 3, "in_top_n_pct": 85.0, "min_rank": 2, "max_rank": 4}},
    ]}}
    table = render_ranking(env)
    assert "| In top 3 (runs) | Rank range | Confidence |" in table
    assert "| 100% | 1 | medium |" in table and "| 85% | 2-4 | low |" in table


def test_drivers_table_and_facts_stay_out_of_the_card():
    env = {**RANKING, "facts": [*RANKING["facts"], {"label": "Drivers: BOS", "value": "strongest: scale", "required": False,
                                                     "ranked": "BOS"}],
           "data": {**RANKING["data"], "ranked": [
               {**RANKING["data"]["ranked"][0], "drivers": {"strongest": {"component": "scale", "percentile": 99.0},
                                                             "weakest": {"component": "long_haul", "percentile": 43.0}}}]}}
    out = render_envelope(env)
    assert ("Drivers (component and national percentile in the peer set):\n\n| Rank | Airport | Strongest | Weakest |\n"
            "|---|---|---|---|\n| 1 | BOS | scale (99) | long haul (43) |") in out
    assert "Drivers: BOS" not in render_facts(env)
