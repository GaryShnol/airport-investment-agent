"""Commentary figure check (P3.6b): hand-written envelopes and commentaries, including real failures
seen in earlier UI tests."""
import pytest

from src.figure_check import (FRAGMENT, check_commentary, finish_commentary, removal_note, restrict_commentary,
                              sentence_problems)

FALLBACK = "No additional commentary beyond the data above."


def f(label, value, **extra):
    return {"label": label, "value": value, "required": False, **extra}


ANC = {"headline": "43.2% of all departures from ANC (incl. cargo) are >= 2,500 mi (38,011 of 87,988); "
                   "passenger service: 11.5% (4,321 of 37,575).",
       "facts": [f("Long-haul share >= 2,500 mi, all flights (including cargo)", "43.2%"),
                 f("Long-haul share >= 2,500 mi, passenger service", "11.5%"),
                 f("Passengers (passenger service)", "2,714,358"),
                 f("Load factor (passenger service, a demand-pressure proxy)", "81.58%"),
                 f("Data period", "2025-01..2025-12")],
       "data": {"airport": "ANC", "name": "Ted Stevens Anchorage International Airport", "weights": {"growth": 0.385}}}

# compare_airports, as in an earlier UI test: SNA's figures were credited to LAX.
LAX_SNA = {"headline": "LAX vs SNA ...",
           "facts": [f("LAX: Departures delayed >= 15 min", "18.1% ± 0.17 pts (189,219 departures)", airport="LAX"),
                     f("SNA: Departures delayed >= 15 min", "13.2% ± 0.37 pts (44,539 departures)", airport="SNA"),
                     f("LAX: Cancelled departures (shown, not scored)", "0.7% ± 0.04 pts", airport="LAX"),
                     f("SNA: Cancelled departures (shown, not scored)", "12.5% ± 0.09 pts", airport="SNA"),
                     f("Verdict: Departures delayed >= 15 min", "LAX higher", metric="dep_delay_rate")],
           "data": {"airports": ["LAX", "SNA"],
                    "names": {"LAX": "Los Angeles International Airport",
                              "SNA": "John Wayne Orange County International Airport"}}}

# New England ranking card: BTV is Burlington, Bradley is BDL.
NEW_ENGLAND = {"headline": "new_england terminal-expansion ranking by score: 1 BOS, 2 BTV, 3 BGR.",
               "facts": [f("#2 BTV (Patrick Leahy Burlington International Airport)", "score 0.676"),
                         f("#6 BDL (Bradley International Airport)", "score 0.49")],
               "data": {"ranked": [{"rank": 2, "iata": "BTV", "name": "Patrick Leahy Burlington International Airport",
                                    "score": 0.676},
                                   {"rank": 6, "iata": "BDL", "name": "Bradley International Airport", "score": 0.49}]}}


@pytest.mark.parametrize("written", ["43.2%", "43.2 %", "43%", "11.5 percent", "2,714,358", "2.7 million",
                                     "2.71 million", "81.6%", "82%", "38.5%", "0.385"])
def test_supported_figures_in_any_format(written):
    assert sentence_problems(f"The figure is {written}.", [ANC]) == []


@pytest.mark.parametrize("written", ["44.2%", "81.7%", "2.8 million", "3,000,000", "13.9%"])
def test_figures_not_in_the_data_are_flagged(written):
    assert sentence_problems(f"The figure is {written}.", [ANC])


def test_years_periods_small_counts_and_codes_are_ignored():
    text = ("In 2025 and 2025-01..2025-12, over 12 months, the top 3 airports of the T-100 data, "
            "an A320 fleet and the 2nd rank.")
    assert sentence_problems(text, [ANC]) == []


def test_sentence_with_an_unsupported_figure_is_removed_and_the_rest_kept():
    out = check_commentary("Cargo drives the gap. About 47% of flights are long haul. Passenger service is 11.5%.",
                           [ANC])
    assert out.text == "Cargo drives the gap. Passenger service is 11.5%."
    assert out.removed == [{"sentence": "About 47% of flights are long haul.", "reasons": ["47% is not in the data"]}]
    assert not out.withheld


def test_nothing_left_gives_the_fallback_line():
    out = finish_commentary(check_commentary("About 47% of flights are long haul.", [ANC]), 4, FALLBACK)
    assert out.withheld and out.text == FALLBACK and len(out.removed) == 1


def test_no_figures_no_change():
    text = "Cargo flights dominate Anchorage's long-haul departures.\n\nAsk about passenger service next."
    out = check_commentary(text, [ANC])
    assert out.text == text and out.removed == []


def test_real_failure_lax_credited_with_snas_figures():
    # Earlier UI test: LAX was credited with SNA's 13.2% and 12.5%.
    out = check_commentary("LAX is more congested: 13.2% of its departures are delayed. LAX also cancels 12.5% "
                           "of its flights. Taxi-out explains part of it.", [LAX_SNA])
    assert out.text == "Taxi-out explains part of it."
    assert [r["reasons"] for r in out.removed] == [["13.2% is SNA's figure, credited to LAX"],
                                                   ["12.5% is SNA's figure, credited to LAX"]]


def test_correctly_attributed_figures_in_one_sentence_pass():
    assert sentence_problems("LAX delays 18.1% of departures, while SNA delays 13.2%.", [LAX_SNA]) == []
    assert sentence_problems("At 13.2%, SNA is lower than LAX.", [LAX_SNA]) == []


@pytest.mark.parametrize("sentence", [
    # lfm2.5:8b in the first eval run (P3.7b): both are correct and must not be removed
    "Its average departures delayed ≥ 15 minutes are 13.2 % versus 18.1 % at LAX.",
    "Although LAX has a higher rate (18.1 % vs 13.2 %), SNA cancels more.",
    "13.2% is SNA's rate, not LAX's.",
])
def test_unclear_attribution_is_not_judged(sentence):
    assert sentence_problems(sentence, [LAX_SNA]) == []


def test_swap_detected_when_both_airports_are_named():
    reasons = sentence_problems("SNA has 18.1% delayed departures and LAX 13.2%.", [LAX_SNA])
    assert reasons == ["18.1% is LAX's figure, credited to SNA", "13.2% is SNA's figure, credited to LAX"]


@pytest.mark.parametrize("sentence", ["Bradley (BTV) ranks second.", "Ranked second, Bradley International (BTV) "
                                      "has a high load factor.", "BTV (Bradley International) ranks second."])
def test_real_failure_btv_called_bradley(sentence):
    reasons = sentence_problems(sentence, [NEW_ENGLAND])
    assert reasons and "is BDL (Bradley International Airport), not BTV" in reasons[0]


@pytest.mark.parametrize("sentence", ["Burlington (BTV) ranks second.", "BTV (Burlington International) is second.",
                                      "Santa Ana (SNA) is not in this card."])
def test_correct_or_unknown_names_pass(sentence):
    assert sentence_problems(sentence, [NEW_ENGLAND]) == []


def test_real_failure_long_haul_share_misread():
    # Earlier UI test: "43.2% of long-haul flights leave Anchorage" (it is 43.2% of ANC's departures).
    reasons = sentence_problems("43.2% of long-haul flights leave Anchorage.", [ANC])
    assert reasons == ["a long-haul share is the share of the airport's departures that are long haul, "
                       "not a share of long-haul flights"]
    assert sentence_problems("43.2% of departures from Anchorage are long haul.", [ANC]) == []


def test_figures_from_earlier_cards_in_the_conversation_count():
    # A follow-up answered from an earlier card: every envelope of the thread is passed.
    assert sentence_problems("BTV scores 0.676 and ANC carries 2,714,358 passengers.", [NEW_ENGLAND, ANC]) == []


def test_lists_and_line_breaks_are_kept():
    out = check_commentary("Points:\n- Passenger service is 11.5%.\n- Cargo is 99.9%.\n- Cargo drives it.", [ANC])
    assert out.text == "Points:\n- Passenger service is 11.5%.\n- Cargo drives it."


# --- restricted mode (P3.8b): rank_airports and compare_airports -------------------------------

NE_RANKING = {"headline": "new_england terminal-expansion ranking by score: 1 BOS, 2 BTV, 3 BGR (10 ranked).",
              "facts": [], "data": {"ranked": [
                  {"rank": 1, "iata": "BOS", "name": "Boston Logan International Airport", "score": 0.799},
                  {"rank": 5, "iata": "HVN", "name": "Tweed New Haven Airport", "score": 0.494},
                  {"rank": 6, "iata": "BDL", "name": "Bradley International Airport", "score": 0.49},
                  {"rank": 10, "iata": "ORH", "name": "Worcester Regional Airport", "score": 0.107}]}}

# lfm2.5:8b in the first eval run: the ranking rewritten as a table, BDL first, a made-up TWE, ORH twice.
REWRITTEN_RANKING = """**Quick Overview of the Data**

| Rank | Airport (IATA) | Score |
|------|----------------|-------|
| 1    | Bradley International Airport (BDL) | 0.49 |
| 3    | Tweed New Haven Airport (TWE) | 0.494 |
| 6    | Worcester Regional (ORH) | 0.107 |
| 7    | Worcester Regional (duplicate entry) | 0.107 |

### Key Take-aways

1. **Bradley (BDL)** leads the set with relatively strong capacity utilisation.
Scores are relative to the national peer set, so they shift when the peer set changes."""


def test_restricted_mode_drops_a_rewritten_ranking_and_keeps_the_caveat():
    out = restrict_commentary(REWRITTEN_RANKING, [NE_RANKING])
    assert out.text == "Scores are relative to the national peer set, so they shift when the peer set changes."
    assert out.mode == "restricted" and not out.hidden
    reasons = {r["sentence"]: r["reasons"] for r in out.removed}
    assert reasons["| 3    | Tweed New Haven Airport (TWE) | 0.494 |"] == ["table, heading or rule (restates the card)"]
    assert reasons["**Bradley (BDL)** leads the set with relatively strong capacity utilisation."] == [
        "names an airport (BDL)"]
    assert reasons["1."] == ["contains a figure"]  # the list marker splits off as its own sentence


def test_restricted_mode_drops_a_backwards_direction():
    # compare card: LAX taxi-out 17.7 min, SNA 16.1 min; the model said LAX's is shorter
    out = restrict_commentary("LAX has a marginally shorter taxi-out, indicating less congestion at the ramp. "
                              "On-Time covers only the large carriers, so smaller airlines are not in these figures.",
                              [LAX_SNA])
    assert out.text == "On-Time covers only the large carriers, so smaller airlines are not in these figures."
    assert out.removed[0]["reasons"] == ["names an airport (LAX)"]


@pytest.mark.parametrize("sentence", ["Los Angeles has the longer taxi-out.", "John Wayne is busier at the ramp.",
                                      "The second airport grows faster than the third."])
def test_restricted_mode_catches_names_without_codes_but_not_unnamed_claims(sentence):
    out = restrict_commentary(sentence, [LAX_SNA, NE_RANKING])
    expected_removed = not sentence.startswith("The second")
    assert bool(out.removed) is expected_removed


# lfm2.5:8b in the fresh-clone run (P5.7): excluded airports named, with a wrong reason (they are
# below the passenger floor, not missing demand context). Excluded entries carry code and name.
NE_WITH_EXCLUDED = {**NE_RANKING, "data": {**NE_RANKING["data"], "excluded": [
    {"iata": "MVY", "name": "Martha's Vineyard Airport", "reason": "87,587 passenger-service passengers, below the 100,000 floor"},
    {"iata": "PSM", "name": "Portsmouth International Airport at Pease",
     "reason": "85,102 passenger-service passengers, below the 100,000 floor"}]}}


@pytest.mark.parametrize("sentence", [
    "Some components (e.g., demand context for MVY, PSM) are missing due to lack of data; these gaps are noted.",
    "Portsmouth and Martha's Vineyard lack demand context data.",
])
def test_restricted_mode_drops_sentences_naming_excluded_airports(sentence):
    caveat = "Scores are relative to the national peer set."
    out = restrict_commentary(f"{sentence} {caveat}", [NE_WITH_EXCLUDED])
    assert out.text == caveat
    assert out.removed[0]["reasons"][-1].startswith("names an airport (")


def test_restricted_mode_with_nothing_left_shows_the_fallback_line():
    out = finish_commentary(restrict_commentary("BOS ranks first with 0.799.", [NE_RANKING]), 4, FALLBACK)
    assert out.text == FALLBACK and out.withheld and not out.hidden and out.removed


def test_restricted_mode_with_nothing_written_hides_the_commentary():
    out = finish_commentary(restrict_commentary("", [NE_RANKING]), 4, FALLBACK)
    assert out.hidden and out.text == "" and not out.withheld and not out.removed


# --- fragments and the removal note (manual test 2026-10-11: Compare BOS and PVD) -------------

BOS_PVD = {"headline": "BOS vs PVD ...", "facts": [f("BOS: Departures delayed >= 15 min", "21.0%", airport="BOS"),
                                                   f("PVD: Departures delayed >= 15 min", "17.4%", airport="PVD")],
           "data": {"airports": ["BOS", "PVD"],
                    "names": {"BOS": "Boston Logan International Airport",
                              "PVD": "Rhode Island T. F. Green International Airport"}}}
# gpt-4o-mini wrote sentences like these; the splitter cut "T. F. Green" and the check left "F.".
BOS_PVD_REPLY = ("The comparison shows that Boston Logan International Airport (BOS) experiences higher congestion. "
                 "In contrast, Rhode Island T. F. Green International Airport (PVD) has a higher cancellation rate. "
                 "The ranking of the two is as follows:\n"
                 "Given the scale difference, the congestion figures compare rates, not totals.")


def test_initials_do_not_end_a_sentence():
    out = restrict_commentary(BOS_PVD_REPLY, [BOS_PVD])
    assert "F." not in out.text and not any(r["sentence"] == "F." for r in out.removed)


def test_fragments_left_after_removals_are_dropped_and_caveats_kept():
    reply = "BOS is busier. The ranking is as follows:\nDelays. Rates compare, totals do not, given the scale gap."
    out = finish_commentary(restrict_commentary(reply, [BOS_PVD]), 4, FALLBACK)
    assert out.text == "Rates compare, totals do not, given the scale gap."
    assert [r["sentence"] for r in out.removed if r["reasons"] == [FRAGMENT]] == ["The ranking is as follows:",
                                                                                 "Delays."]


def test_real_bos_pvd_reply_keeps_only_the_caveat():
    out = finish_commentary(restrict_commentary(BOS_PVD_REPLY, [BOS_PVD]), 4, FALLBACK)
    assert out.text == "Given the scale difference, the congestion figures compare rates, not totals."


def test_untouched_commentary_is_never_trimmed():
    out = finish_commentary(check_commentary("Cargo drives it", [ANC]), 4, FALLBACK)
    assert out.text == "Cargo drives it" and not out.removed


def test_june_july_case_survives_the_fragment_step():
    real = ("The excess is concentrated on key transcontinental routes (EWR, JFK, ORD account for a large share "
            "of the spill) and peak June–July and December months drive the pressure.")
    out = finish_commentary(check_commentary(f"{real} Load peaked at 99.9%.", [SFO_GAP]), 4, FALLBACK)
    assert out.text == real and len(out.removed) == 1


def test_removal_note_counts_sentences_and_lists_each_reason_once():
    removed = [{"sentence": "a", "reasons": ["names an airport (BOS)"]},
               {"sentence": "b", "reasons": ["contains a figure", "names an airport (BOS, PVD)"]},
               {"sentence": "c", "reasons": ["names an airport (PVD)"]}, {"sentence": "F.", "reasons": [FRAGMENT]}]
    assert removal_note(removed, "{count} removed ({reasons}).") == (
        f"4 removed (names an airport; contains a figure; {FRAGMENT}).")
    assert removal_note([], "{count} removed ({reasons}).") == ""


# --- periods (P5.9b) -------------------------------------------------------------------------

SFO_GAP = {"headline": "PROXY, not a measurement: SFO load factor was above the 80% target in 10 of 12 months "
                       "in 2025-01..2025-12 (peak 2025-06 89.4%).",
           "facts": [f("Months with load factor above 80%", "10 of 12"),
                     f("Peak months (load factor)", "2025-06 89.4%, 2025-07 84.8%, 2025-12 84.0%"),
                     f("Passenger growth vs 2024-01..2024-12", "+4.7%"),
                     f("Freshness", "stale (upstream has 2026-07 for ontime)")],
           "data_period": "2025-01..2025-12"}


def test_period_check_drops_the_real_sfo_failure():
    # lfm2.5:8b in the fresh-clone run (P5.7): the window is 2025-01..2025-12, not March to December
    row = ("| **Load factor above 80 %** | In 10 of the 12 months (Mar 2025 – Dec 2025) SFO’s load factor "
           "was > 80 %, peaking at 89.4 % in June and still high in several other months. |")
    out = check_commentary(f"{row}\nDemand presses hardest in summer.", [SFO_GAP])
    assert "Mar 2025" not in out.text and out.text.endswith("Demand presses hardest in summer.")
    assert out.removed[0]["reasons"] == ["'Mar 2025 – Dec 2025' is not the data period"]


@pytest.mark.parametrize("sentence", [
    "The data covers January to December 2025.",
    "The window is 2025-01..2025-12.",
    "Jan–Dec 2025 shows a steady load.",
    "Load factor peaked in June 2025.",
    "It peaked in 2025-06 and stayed high in July.",
    "Passengers grew compared with 2024.",
    "Upstream already has 2026-07.",
    "Growth is measured against 2024-01..2024-12.",
])
def test_period_check_accepts_periods_the_data_covers(sentence):
    assert check_commentary(sentence, [SFO_GAP]).removed == []


@pytest.mark.parametrize("sentence, reason", [
    ("Load factor stayed above target from March to December.", "'March to December' is not the data period"),
    ("The window is 2025-03..2025-12.", "'2025-03..2025-12' is not the data period"),
    ("Load factor peaked in June 2023.", "'June 2023' is outside the data period"),
    ("In 2022 the airport was less busy.", "2022 is outside the data period"),
])
def test_period_check_drops_periods_the_data_does_not_cover(sentence, reason):
    out = check_commentary(sentence, [SFO_GAP])
    assert out.removed and out.removed[0]["reasons"] == [reason]


def test_month_names_from_the_peak_months_pass():
    # gpt-5-mini, q4 (eval run 20261010-215548): removed as "'June–July' is not the data period"
    real = ("The excess is concentrated on key transcontinental routes (EWR, JFK, ORD account for a large share "
            "of the spill) and peak June–July and December months drive the pressure.")
    assert check_commentary(real, [SFO_GAP]).removed == []


@pytest.mark.parametrize("sentence", [
    "Load peaked in June-July 2025.",
    "June to July and December carry the peak.",
    "Jun–Jul is the summer peak.",
])
def test_month_ranges_named_in_the_facts_pass(sentence):
    assert check_commentary(sentence, [SFO_GAP]).removed == []


@pytest.mark.parametrize("sentence, reason", [
    ("Load stayed high from June to December.", "'June to December' is not the data period"),
    ("Load peaked in June-July 2023.", "'June-July 2023' is not the data period"),
    ("Peak months were May-June.", "'May-June' is not the data period"),
])
def test_month_ranges_not_named_or_outside_the_data_still_drop(sentence, reason):
    out = check_commentary(sentence, [SFO_GAP])
    assert out.removed and out.removed[0]["reasons"] == [reason]


def test_period_check_needs_a_data_period():
    assert check_commentary("Load factor peaked in June 2023.", [{"headline": "x", "facts": []}]).removed == []


# --- airport aliases and month counts (audit H3, H7) ------------------------------------------

# compare_airports as the tool now returns it: the city each airport serves, "LA" and the user's words.
LAX_SNA_ALIASES = {**LAX_SNA, "data": {**LAX_SNA["data"],
                                       "aliases": {"LAX": ["Los Angeles", "LA"], "SNA": ["Santa Ana"]}}}


@pytest.mark.parametrize("sentence", [
    "Santa Ana has the longer taxi-out time.",          # false (LAX 17.7 vs SNA 16.1 min): passed before
    "santa ana is the more congested airport.",
    "LA is less delayed overall.",
    "Additionally, Santa Ana has a higher cancellation rate.",   # kept in the audit's live Q2 run
])
def test_restricted_mode_catches_cities_and_aliases(sentence):
    caveat = "On-Time covers only the large carriers."
    out = restrict_commentary(f"{sentence} {caveat}", [LAX_SNA_ALIASES])
    assert out.text == caveat
    assert out.removed[0]["reasons"][-1].startswith("names an airport (")


@pytest.mark.parametrize("sentence", [
    "These are per-flight rates, so the larger airport is not penalized for its size.",
    "La plupart des retards sont mineurs.",     # "la" is not the alias "LA"
    "A Santa Claus rally is not in the data.",   # a partial city name is not the city
])
def test_restricted_mode_aliases_do_not_drop_unnamed_caveats(sentence):
    assert restrict_commentary(sentence, [LAX_SNA_ALIASES]).removed == []


def test_swapped_figure_credited_by_city_name_is_caught():
    # 13.2% is SNA's delayed-departure share; credited to LAX by its city
    assert sentence_problems("Los Angeles has 13.2% of departures delayed.", [LAX_SNA_ALIASES]) == [
        "13.2% is SNA's figure, credited to LAX"]
    assert sentence_problems("Santa Ana has 13.2% of departures delayed.", [LAX_SNA_ALIASES]) == []


@pytest.mark.parametrize("sentence", [
    "SFO was above the target in 11 of 12 months.",       # the card says 10 of 12 (audit H7)
    "Load factor was above 80% in 12 out of 12 months.",
    "It ran hot in 9 of the 12 months.",
])
def test_month_counts_not_in_the_data_are_removed(sentence):
    out = check_commentary(sentence, [SFO_GAP])
    assert out.removed and out.removed[0]["reasons"][-1].endswith("months' is not in the data")


@pytest.mark.parametrize("sentence", [
    "SFO was above the target in 10 of 12 months.",
    "Load factor topped 80% in 10 out of the 12 months, peaking June–July.",   # "June–July" stays
    "Peak months were June–July and December, 10 of 12 months above target.",
])
def test_month_counts_from_the_data_pass_and_june_july_stays(sentence):
    assert check_commentary(sentence, [SFO_GAP]).removed == []


# --- airport names next to codes (fix/entity-name-match) ---------------------------------------

# Names as in the airports table. Real failure: "At San Francisco International Airport (SFO)" was
# called PSM, because "at" (from the sentence) is a word of "Portsmouth International Airport at Pease".
NAMED = {"data": {"names": {"SFO": "San Francisco International Airport", "LAX": "Los Angeles International Airport",
                            "SNA": "John Wayne Orange County International Airport",
                            "PWM": "Portland International Jetport", "PSM": "Portsmouth International Airport at Pease",
                            "OAK": "Oakland San Francisco Bay Airport"}}}


@pytest.mark.parametrize("sentence", [
    "At San Francisco International Airport (SFO) load factor stays high.",
    "SFO (San Francisco International Airport) is above the target.",
    "Los Angeles International Airport (LAX) is busier.",
    "At Los Angeles International (LAX) delays are longer.",
    "John Wayne Orange County International Airport (SNA) cancels more.",
    "John Wayne (SNA) cancels more.",
    "SNA (Orange County) cancels more.",
    "Portland International Jetport (PWM) is in Maine.",
    "Portsmouth International Airport at Pease (PSM) is below the floor.",
])
def test_official_and_short_names_resolve_to_their_own_code(sentence):
    assert sentence_problems(sentence, [NAMED]) == []


@pytest.mark.parametrize("sentence, reason", [
    ("Portland International Jetport (PSM) is in Maine.",
     "'Portland International Jetport' is PWM (Portland International Jetport), not PSM"),
    ("John Wayne (LAX) cancels more.", "'John Wayne' is SNA (John Wayne Orange County International Airport), not LAX"),
    ("San Francisco International Airport (OAK) is busier.",
     "'San Francisco International Airport' is SFO (San Francisco International Airport), not OAK"),
])
def test_genuine_name_mismatch_is_removed_with_the_right_note(sentence, reason):
    out = check_commentary(f"{sentence} Demand presses hardest in summer.", [NAMED])
    assert out.text == "Demand presses hardest in summer."
    assert out.removed[0]["reasons"][0].startswith(reason)
