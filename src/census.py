"""Census Bureau access for ingest only: metro population (Data API) and airport -> CBSA (Geocoder).

Network calls go through a `session` argument (a `requests.Session` in production, a stub in
tests). Responses are parsed by pure functions. The API key travels in the query string, so no
error message ever includes a URL: exceptions are reported by type and the key is redacted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

# Census API geography name for CBSAs (metropolitan and micropolitan statistical areas).
CBSA_GEOGRAPHY = "metropolitan statistical area/micropolitan statistical area"
# Geocoder layers holding CBSAs, and the kind recorded for each.
GEOCODER_LAYERS = {"Metropolitan Statistical Areas": "metro", "Micropolitan Statistical Areas": "micro"}
USER_AGENT = "airport-investment-agent ingest"
# Required wherever Census data is shown (Census Data API terms): sidebar, README, ingest output.
ATTRIBUTION = "This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau."


class CensusError(RuntimeError):
    """A Census request failed; the message is safe to print (no URL, no key)."""


def redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


def _html_title(text: str) -> str | None:
    match = re.search(r"<title>\s*(.*?)\s*</title>", text, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else None


def parse_population(status: int, text: str, variable: str) -> list[tuple[str, str, int]]:
    """(cbsa, name, population) rows from a Census Data API response for CBSA_GEOGRAPHY.

    The API answers a missing or invalid key with HTTP 200 and an HTML page, so the body is
    checked, not only the status. Values come back as strings; negative values are Census
    annotation codes (no estimate) and are skipped."""
    if status != 200:
        title = _html_title(text)
        raise CensusError(f"Census API returned HTTP {status}" + (f" ({title})" if title else ""))
    try:
        rows = json.loads(text)
    except ValueError:
        title = _html_title(text) or "not JSON"
        hint = " Set a valid CENSUS_API_KEY in .env." if "key" in title.lower() else ""
        raise CensusError(f"Census API returned a page instead of data ({title}).{hint}") from None
    if not isinstance(rows, list) or len(rows) < 2 or not isinstance(rows[0], list):
        raise CensusError("Census API returned no rows")
    header = rows[0]
    missing = [c for c in ("NAME", variable, CBSA_GEOGRAPHY) if c not in header]
    if missing:
        raise CensusError(f"Census API response lacks columns {missing}")
    i_name, i_value, i_geo = (header.index(c) for c in ("NAME", variable, CBSA_GEOGRAPHY))
    out = []
    for row in rows[1:]:
        try:
            population = int(row[i_value])
        except (TypeError, ValueError):
            continue
        if population >= 0:
            out.append((str(row[i_geo]), str(row[i_name]), population))
    if not out:
        raise CensusError("Census API returned no usable population values")
    return out


def fetch_population(session, api_url: str, dataset: str, variable: str, vintage: int,
                     key: str | None, timeout: float) -> list[tuple[str, str, int]]:
    """Population of every CBSA for one vintage. Raises CensusError with a printable message."""
    params = {"get": f"NAME,{variable}", "for": f"{CBSA_GEOGRAPHY}:*"}
    if key:
        params["key"] = key
    try:
        response = session.get(f"{api_url}/{vintage}/{dataset}", params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise CensusError(f"Census API not reachable ({type(exc).__name__})") from None
    try:
        return parse_population(response.status_code, response.text, variable)
    except CensusError as exc:
        raise CensusError(f"vintage {vintage}: {redact(str(exc), key)}") from None


@dataclass(frozen=True)
class GeocodeResult:
    """One airport's CBSA, or None when it lies outside every metro and micro area."""
    cbsa: str | None
    name: str | None
    kind: str | None    # metro | micro


def parse_geocoder(payload: dict) -> GeocodeResult:
    """The CBSA from a Geocoder `geographies/coordinates` response; metro wins over micro (an area
    is one or the other, so both never appear for one point). Raises ValueError on an unexpected shape."""
    geographies = payload["result"]["geographies"]
    for layer, kind in GEOCODER_LAYERS.items():
        for area in geographies.get(layer) or []:
            code = area.get("CBSA") or area.get("GEOID")
            if code:
                return GeocodeResult(str(code), area.get("NAME"), kind)
    return GeocodeResult(None, None, None)


def geocode(session, geocoder_url: str, benchmark: str, vintage: str, lat: float, lon: float,
            timeout: float) -> GeocodeResult:
    """CBSA containing a point. Raises CensusError when the request or the response fails
    (the caller retries such airports on the next run; a point outside any CBSA is a result)."""
    params = {"x": lon, "y": lat, "benchmark": benchmark, "vintage": vintage,
              "layers": ",".join(GEOCODER_LAYERS), "format": "json"}
    try:
        response = session.get(geocoder_url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise CensusError(f"Geocoder not reachable ({type(exc).__name__})") from None
    if response.status_code != 200:
        raise CensusError(f"Geocoder returned HTTP {response.status_code}")
    try:
        return parse_geocoder(response.json())
    except (ValueError, KeyError, TypeError):
        raise CensusError("Geocoder returned an unexpected response") from None


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s
