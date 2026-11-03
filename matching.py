
"""Exact, auditable company matching for the AIM AQ portfolio tool.

This module reads the existing company dataset. It never edits or infers
AQ classifications and never automatically assigns partial/fuzzy matches.
"""

import re
from collections import defaultdict

import pandas as pd


VALID_BANDS = {"Green", "Orange", "Red"}

COMPANY_COL = "Company Name"
ISIN_COL = "ISIN Code"
SYMBOL_COL = "Symbol"
BAND_COL = "Category"


def _clean(value):
    """Convert a value to a stripped string; treat missing values as empty."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalise_identifier(value):
    """Normalise ISINs and symbols for exact matching."""
    return re.sub(r"\s+", "", _clean(value)).upper()


def normalise_name(value):
    """Normalise company names without using partial or fuzzy matching."""
    name = _clean(value).casefold()

    # Ignore common legal-form words when comparing company names.
    name = re.sub(r"\b(?:limited|ltd|private|pvt)\b", " ", name)

    # Treat punctuation and repeated whitespace consistently.
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return " ".join(name.split())


def build_match_index(companies_df):
    """Build exact-match indexes and report collisions in the dataset."""
    required = {COMPANY_COL, ISIN_COL, SYMBOL_COL, BAND_COL}
    missing = required - set(companies_df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    records = []
    isin_index = defaultdict(list)
    symbol_index = defaultdict(list)
    name_index = defaultdict(list)

    for row_number, row in enumerate(
        companies_df.to_dict(orient="records"), start=2
    ):
        record = {
            "company": _clean(row.get(COMPANY_COL)),
            "isin": normalise_identifier(row.get(ISIN_COL)),
            "symbol": normalise_identifier(row.get(SYMBOL_COL)),
            "band": _clean(row.get(BAND_COL)),
            "source_row": row_number,
        }

        records.append(record)

        if record["isin"]:
            isin_index[record["isin"]].append(record)

        if record["symbol"]:
            symbol_index[record["symbol"]].append(record)

        normalised = normalise_name(record["company"])
        if normalised:
            name_index[normalised].append(record)

    collisions = {
        "isin": {
            key: [r["company"] for r in rows]
            for key, rows in isin_index.items()
            if len(rows) > 1
        },
        "symbol": {
            key: [r["company"] for r in rows]
            for key, rows in symbol_index.items()
            if len(rows) > 1
        },
        "name": {
            key: [r["company"] for r in rows]
            for key, rows in name_index.items()
            if len(rows) > 1
        },
    }

    return {
        "records": records,
        "isin": dict(isin_index),
        "symbol": dict(symbol_index),
        "name": dict(name_index),
        "collisions": collisions,
    }


def _result(record=None, match_type="none", status="", warnings=None):
    """Return a consistent result structure for every matching outcome."""
    warnings = warnings or []

    valid_record = (
        record is not None
        and record["band"] in VALID_BANDS
    )

    if record is None:
        company = ""
        band = ""
    else:
        company = record["company"]
        band = record["band"] if valid_record else ""

        if not valid_record:
            warnings.append(
                "Dataset classification is missing or invalid; "
                "holding remains unclassified."
            )
            status = "Needs review — invalid or missing AQ band"

    return {
        "matched_company": company,
        "band": band,
        "match_type": match_type,
        "status": status,
        "warnings": warnings,
    }


def match_holding(name, isin, symbol, match_index):
    """Match one holding using exact ISIN, symbol, then company name.

    Parameters
    ----------
    name, isin, symbol:
        Values supplied for one portfolio holding.
    match_index:
        Index returned by build_match_index(companies_df).
    """
    name = _clean(name)
    isin_key = normalise_identifier(isin)
    symbol_key = normalise_identifier(symbol)
    name_key = normalise_name(name)

    warnings = []

    if not any((name_key, isin_key, symbol_key)):
        return _result(
            status="Missing identifiers",
            warnings=["No company name, ISIN or symbol supplied."],
        )

    # 1. Try exact ISIN.
    if isin_key:
        candidates = match_index["isin"].get(isin_key, [])

        if len(candidates) == 1:
            record = candidates[0]

            if name_key and name_key != normalise_name(record["company"]):
                warnings.append(
                    "Uploaded company name differs from the dataset name; "
                    "verify this holding."
                )
                return _result(
                    record,
                    "isin",
                    "Assessed — name differs: review",
                    warnings,
                )

            return _result(
                record,
                "isin",
                "Assessed — ISIN match",
                warnings,
            )

        if len(candidates) > 1:
            return _result(
                status="Ambiguous — needs review",
                warnings=[
                    "ISIN occurs more than once in the dataset; "
                    "no AQ band assigned."
                ],
            )
        else:
            warnings.append(
                "ISIN not found in the dataset; any alternative match "
                "requires review."
            )

    # 2. Try exact symbol if ISIN did not uniquely identify a company.
    if symbol_key:
        candidates = match_index["symbol"].get(symbol_key, [])

        if len(candidates) == 1:
            record = candidates[0]

            if name_key and name_key != normalise_name(record["company"]):
                warnings.append(
                    "Uploaded company name differs from the dataset name; "
                    "verify this holding."
                )

            if isin_key:
                status = "Assessed — symbol match; review ISIN"
            else:
                status = "Assessed — symbol match"

            if name_key and name_key != normalise_name(record["company"]):
                status = "Assessed — name differs: review"

            return _result(record, "symbol", status, warnings)

        if len(candidates) > 1:
            return _result(
                status="Ambiguous — needs review",
                warnings=[
                    "Symbol occurs more than once in the dataset; "
                    "no AQ band assigned."
                ],
            )
        elif symbol_key:
            warnings.append("Symbol not found in the dataset.")

    # 3. Try exact normalised company name. Never use substring matching.
    if name_key:
        candidates = match_index["name"].get(name_key, [])

        if len(candidates) == 1:
            record = candidates[0]

            if isin_key or symbol_key:
                status = "Assessed — exact name match; review identifiers"
            else:
                status = "Assessed — exact name match"

            return _result(
                record,
                "exact_name",
                status,
                warnings,
            )

        if len(candidates) > 1:
            warnings.append(
                "Normalised company name matches multiple dataset records; "
                "no AQ band assigned."
            )
            return _result(
                status="Ambiguous — needs review",
                warnings=warnings,
            )

    return _result(
        status="Not currently assessed",
        warnings=warnings or [
            "No exact ISIN, symbol or company-name match found."
        ],
    )


def get_collision_report(match_index):
    """Return duplicate-key details without changing the dataset."""
    return match_index["collisions"]
