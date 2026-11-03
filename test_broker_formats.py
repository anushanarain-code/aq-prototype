import pandas as pd

from broker_formats import detect_broker_format, normalise_portfolio
from matching import build_match_index, match_holding
companies_df = pd.read_csv("data/aim_companies.csv")
match_index = build_match_index(companies_df)

zerodha = pd.DataFrame({
    "Instrument": ["BAJAJ-AUTO"],
    "isin": ["INE917I01010"],
    "Qty.": [10],
    "Cur. val": [100000],
})

groww = pd.DataFrame({
    "Stock Name": ["Bajaj Auto"],
    "ISIN": ["INE917I01010"],
    "Qty": [10],
    "Current Value": [100000],
})

icici = pd.DataFrame({
    "Scrip Name": ["Bajaj Auto"],
    "ISIN Code": ["INE917I01010"],
    "Net Quantity": [10],
    "Mkt Value": [100000],
})


for name, data in [
    ("Zerodha", zerodha),
    ("Groww", groww),
    ("ICICI Direct", icici),
]:
    broker, mapping = detect_broker_format(data)
    print("\nBroker:", name, "=>", broker)

    if broker:
        standard_df = normalise_portfolio(data, mapping)
        print(standard_df)

        for _, holding in standard_df.iterrows():
            result = match_holding(
                holding["Company Name"],
                holding["ISIN"],
                holding["Symbol"],
                match_index
            )
            print("AIM match:", result)

        print("---")