import re
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="AIM Ahimsa Quotient",
    page_icon="🌱",
    layout="centered",
)


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

DATA_FILE = Path(__file__).parent / "data" / "aq_master_list.csv"

EXPECTED_COLUMNS = [
    "Company Name",
    "Symbol",
    "ISIN Code",
    "Macro Sector",
    "Sector",
    "Industry",
    "Basic Industry",
    "Category",
    "Justification",
]


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error("AQ dataset not found.")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            f"The AQ dataset is missing these columns: {missing_columns}"
        )
        st.stop()

    df = df[EXPECTED_COLUMNS].copy()

    for column in EXPECTED_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()

    return df


df = load_data()


# ---------------------------------------------------------
# Matching
# ---------------------------------------------------------

def normalise_text(text):
    """Make company names easier to compare."""
    text = str(text).lower().strip()

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def find_matches(company_input):
    """
    Find companies in the dataset using deterministic matching.

    1. Try exact normalized company-name match.
    2. If there is no exact match, try a conservative partial match.
    """

    query = normalise_text(company_input)

    if not query:
        return pd.DataFrame()

    normalized_names = df["Company Name"].apply(normalise_text)

    # Exact match
    exact_matches = df[normalized_names == query]

    if not exact_matches.empty:
        return exact_matches

    # Conservative partial match
    partial_mask = normalized_names.apply(
        lambda name: query in name or name in query
    )

    return df[partial_mask]


# ---------------------------------------------------------
# Company result
# ---------------------------------------------------------

def display_company_result(row):

    st.divider()

    st.subheader(row["Company Name"])

    band = row["Category"].strip().upper()

    st.markdown(f"### AQ Band: **{band}**")

    st.markdown("#### Company information")

    fields = [
        ("Symbol", row["Symbol"]),
        ("ISIN Code", row["ISIN Code"]),
        ("Macro Sector", row["Macro Sector"]),
        ("Sector", row["Sector"]),
        ("Industry", row["Industry"]),
        ("Basic Industry", row["Basic Industry"]),
    ]

    for label, value in fields:
        if value:
            st.write(f"**{label}:** {value}")

    st.markdown("#### AIM justification")

    if row["Justification"]:
        st.write(row["Justification"])
    else:
        st.write("No justification is available in the dataset.")

    st.caption(
        "Source: AIM AQ dataset provided for this prototype."
    )


# ---------------------------------------------------------
# Not assessed
# ---------------------------------------------------------

def display_not_assessed():

    st.warning(
        "We couldn't find an AQ assessment for this company "
        "in the current dataset."
    )

    st.write(
        "This does not mean that the company has a particular "
        "AQ classification."
    )


# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------

def home_page():

    st.title("AIM Ahimsa Quotient")

    st.write(
        "Check companies against AIM's Ahimsa Quotient "
        "and explore their AQ classification."
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Check a company",
            use_container_width=True,
        ):
            st.session_state["page"] = "company"
            st.rerun()

    with col2:
        if st.button(
            "Assess a portfolio",
            use_container_width=True,
        ):
            st.session_state["page"] = "portfolio"
            st.rerun()

    st.write("")

    with st.expander("What is AQ?"):
        st.write(
            "The Ahimsa Quotient (AQ) is AIM's framework for "
            "classifying companies according to their relationship "
            "with animal harm, based on the AIM AQ dataset used "
            "for this prototype."
        )


# ---------------------------------------------------------
# Company search page
# ---------------------------------------------------------

def company_page():

    if st.button("← Back to home"):
        st.session_state["page"] = "home"
        st.rerun()

    st.title("Check a company")

    company_input = st.text_input(
        "Company name",
        placeholder="Enter a company name",
    )

    if st.button(
        "Check company",
        type="primary",
        use_container_width=True,
    ):

        if not company_input.strip():
            st.warning("Please enter a company name.")
            return

        matches = find_matches(company_input)

        if len(matches) == 1:

            display_company_result(matches.iloc[0])

        elif len(matches) > 1:

            st.info("We found multiple possible matches.")

            options = matches["Company Name"].tolist()

            selected_company = st.selectbox(
                "Select the company you mean",
                options,
            )

            selected_row = matches[
                matches["Company Name"] == selected_company
            ].iloc[0]

            display_company_result(selected_row)

        else:

            display_not_assessed()


# ---------------------------------------------------------
# Portfolio page
# ---------------------------------------------------------

def portfolio_page():

    if st.button("← Back to home"):
        st.session_state["page"] = "home"
        st.rerun()

    st.title("Assess a portfolio")

    st.write(
        "Enter one company name per line."
    )

    portfolio_input = st.text_area(
        "Companies",
        placeholder=(
            "Company A\n"
            "Company B\n"
            "Company C"
        ),
        height=200,
    )

    if st.button(
        "Assess portfolio",
        type="primary",
        use_container_width=True,
    ):

        companies = [
            line.strip()
            for line in portfolio_input.splitlines()
            if line.strip()
        ]

        if not companies:
            st.warning("Please enter at least one company.")
            return

        results = []

        for company in companies:

            matches = find_matches(company)

            if len(matches) == 1:

                row = matches.iloc[0]

                results.append({
                    "User input": company,
                    "Matched company": row["Company Name"],
                    "AQ Band": row["Category"],
                    "Status": "Matched",
                })

            else:

                results.append({
                    "User input": company,
                    "Matched company": "",
                    "AQ Band": "",
                    "Status": "Not currently assessed",
                })

        results_df = pd.DataFrame(results)

        st.subheader("Portfolio results")

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
        )

        # ---------------------------------------------
        # Holding counts
        # ---------------------------------------------

        st.subheader("Holding counts")

        matched_results = results_df[
            results_df["Status"] == "Matched"
        ]

        green_count = (
            matched_results["AQ Band"]
            .eq("Green")
            .sum()
        )

        orange_count = (
            matched_results["AQ Band"]
            .eq("Orange")
            .sum()
        )

        red_count = (
            matched_results["AQ Band"]
            .eq("Red")
            .sum()
        )

        not_assessed_count = (
            results_df["Status"]
            .eq("Not currently assessed")
            .sum()
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Green", green_count)
        col2.metric("Orange", orange_count)
        col3.metric("Red", red_count)
        col4.metric("Not assessed", not_assessed_count)

        # ---------------------------------------------
        # Optional company details
        # ---------------------------------------------

        if not matched_results.empty:

            st.subheader("View company details")

            selected_company = st.selectbox(
                "Select a matched company",
                matched_results["Matched company"].tolist(),
            )

            selected_row = df[
                df["Company Name"] == selected_company
            ].iloc[0]

            display_company_result(selected_row)


# ---------------------------------------------------------
# Routing
# ---------------------------------------------------------

if "page" not in st.session_state:
    st.session_state["page"] = "home"


if st.session_state["page"] == "home":

    home_page()

elif st.session_state["page"] == "company":

    company_page()

elif st.session_state["page"] == "portfolio":

    portfolio_page()