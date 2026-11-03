import pandas as pd


STANDARD_COLUMNS = [
    "Company Name",
    "ISIN",
    "Symbol",
    "Quantity",
    "Market Value",
]


def normalise_portfolio(df, column_map):
    """
    Convert a broker portfolio into AIM's standard column structure.
    """

    result = pd.DataFrame(index=df.index, columns=STANDARD_COLUMNS)

    for standard_name, source_name in column_map.items():
        if source_name and source_name in df.columns:
            result[standard_name] = df[source_name]

    return result
BROKER_FORMATS = {
    "Zerodha": {
        "Company Name": ["Company Name", "company_name"],
        "ISIN": ["isin", "ISIN"],
        "Symbol": ["tradingsymbol", "Trading Symbol", "Symbol", "Instrument"],
        "Quantity": ["quantity", "Quantity", "Qty."],
        "Market Value": ["market value", "Market Value", "Cur. val", "Current value"],
    },

    "Groww": {
        "Company Name": ["Company Name", "Stock Name", "name"],
        "ISIN": ["ISIN", "isin"],
        "Symbol": ["Symbol", "Trading Symbol", "Trading symbol"],
        "Quantity": ["Quantity", "Qty", "quantity"],
        "Market Value": ["Current Value", "Market Value", "Invested Value"],
    },

    "ICICI Direct": {
        "Company Name": ["Scrip Name"],
        "ISIN": ["ISIN Code"],
        "Symbol": ["Scrip Code"],
        "Quantity": ["Net Quantity"],
        "Market Value": ["Mkt Value"],
    },
}

def detect_broker_format(df):
    columns = {str(col).strip().lower(): col for col in df.columns}

    matches = []

    for broker, fields in BROKER_FORMATS.items():
        matched = {}

        for standard_name, candidates in fields.items():
            for candidate in candidates:
                source = columns.get(candidate.lower())

                if source is not None:
                    matched[standard_name] = source
                    break

        if (
            matched.get("Company Name") or matched.get("Symbol")
        ) and matched.get("ISIN") and (
            matched.get("Quantity") or matched.get("Market Value")
        ):
            matches.append((broker, matched))

    if len(matches) == 1:
        return matches[0]

    return None, {}
    columns = {str(col).strip().lower(): col for col in df.columns}

    for broker, fields in BROKER_FORMATS.items():
        matched = {}

        for standard_name, candidates in fields.items():
            for candidate in candidates:
                source = columns.get(candidate.lower())

                if source is not None:
                    matched[standard_name] = source
                    break

        if matched.get("Symbol") and (
            matched.get("Quantity") or matched.get("Market Value")
        ):
            return broker, matched

    return None, {}