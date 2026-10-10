"""Commentary figure check (P3.6b): hand-written envelopes and commentaries, including real failures
seen in earlier UI tests."""
import pytest

from src.figure_check import WITHHELD, check_commentary, sentence_problems


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


def test_nothing_left_gives_the_fixed_note():
    out = check_commentary("About 47% of flights are long haul.", [ANC])
    assert out.withheld and out.text == WITHHELD


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
