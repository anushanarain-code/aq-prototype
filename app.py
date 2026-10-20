import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="AIM Ahimsa Quotient",
    page_icon="🌱",
    layout="centered",
)


DATA_FILE = Path(__file__).parent / "data" / "aim_companies.csv"

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

def load_aim_methodology():
    methodology_path = Path("aim_methodology.txt")

    if not methodology_path.exists():
        return ""

    return methodology_path.read_text(
        encoding="utf-8"
    ).strip()
def build_ai_prompt(question, context=""):
    methodology = get_relevant_methodology(question)

    return f"""
You are an assistant for AIM's Ahimsa Quotient (AQ).

Answer the user's question using ONLY the AIM methodology and
company information supplied below.

STRICT RULES:
- The supplied AIM AQ dataset is authoritative for a company's AQ band.
- Never independently classify a company.
- Never change, override, or infer an AQ band.
- Never invent company activities, products, revenue, animal links,
  ethical practices, sustainability claims, or external sources.
- Never introduce a numerical AQ score.
- Do not use general model knowledge to fill missing information.
- For methodology questions, answer only from the supplied AIM methodology.
  Do not use any company records to create examples or explanations unless
  the user explicitly asks about a company.
- If the supplied methodology directly answers the question, do not add
  examples, company claims, section numbers, or other information that is
  not present in the supplied methodology.
- Do not generalize a specific AIM rule into a broader principle.
- If the methodology gives a specific classification rule, state only that
  rule and do not extend it to cases not explicitly covered.
- For a question about a specific methodology rule, if the supplied
  methodology gives the answer directly, answer with only that rule.
  Do not add an explanation, rationale, interpretation, or application of
  other methodology rules unless the supplied methodology explicitly gives
  that explanation for the specific rule.   
- If the answer is established by the supplied AIM methodology or
  company/dataset information, answer it directly.
- If the supplied information does not establish the answer, say:
  "The available AIM information does not establish this."
- Keep the answer clear and concise, in 2–4 sentences.
- Do not simply repeat the AIM Justification.
- Explain what the supplied justification means in terms of the AIM
  methodology.
- Preserve the meaning and precision of the supplied justification;
  do not turn a specific statement into a broader factual claim.
- Do not describe a minority, smaller, allied, incidental, or limited activity as "core" unless the supplied company information explicitly says it is core or a large share of the business.
- Focus on the animal-harm relevance of the supplied information,
  rather than treating the company's ordinary business activity as
  the reason for its AQ band.
- Do not invent section numbers, citations, source names, URLs, or references.
- Never mention section numbers such as "Section 4" or "Section 7" in the answer, even if they appear in the supplied company justification.  

RELEVANT AIM METHODOLOGY:
{methodology}

SUPPLIED COMPANY/DATASET INFORMATION:
{context}

USER QUESTION:
{question}
""".strip()

def get_relevant_methodology(question):
    question = normalise_text(question)

    sections = {
        "bands": """
THREE AQ BANDS

GREEN — HIGH AQ
Minimal or no involvement in animal harm.

ORANGE — MEDIUM AQ
A genuine but smaller animal-linked involvement, or finance activity
where money is spread across a wide range of industries.

RED — LOW AQ
Direct, significant, or core involvement in animal harm.

The colour band is a broad indication. It is not an exact numerical
measurement.
""",

        "core_minority": """
CORE VS MINORITY TEST

Core or large share:
Animal product or activity is the heart of the business or generates
a large share of revenue → Red.

Real but smaller line:
Genuine animal-linked product or activity exists but is a minority
part of an otherwise animal-neutral business → Orange.

Tiny, replaceable or far-fetched:
The link is very small, easily replaced by a non-animal option, or only
a distant possibility → Green.
""",

        "process": """
CLASSIFICATION PROCESS

1. Examine the company's actual business, products, raw materials,
business segments and revenue mix.

2. Determine whether the company's own animal involvement is a real
and significant part of the business or only a small or far-fetched link.

3. Check the share of relevant activity and apply the framework rules.

4. Assign Red, Orange or Green and document a factual explanation.

5. Peer-check classifications against similar companies and revisit
classifications when new information becomes available.
""",

        "finance": """
FINANCE COMPANIES

Finance companies are judged according to what their money ultimately
supports.

Finance companies whose money is spread across a wide mix of industries,
including animal-based businesses, are classified Orange.

Money restricted to one animal-neutral purpose, or pure market
infrastructure that does not itself direct capital into animal
businesses, can be classified Green.
""",

        "special_rules": """
SPECIAL RULES

Products specifically created for animal-rearing or animal-processing
industries can affect the AQ classification.

Products designed to kill insects or rodents are classified as Red.

Core pharmaceutical and cosmetic makers are classified as Red under
AIM's stated animal-testing rule.

Hospitals that merely use medicines remain Green under the framework.

Synthetic substitutes for animal-derived materials are treated as
Green under the framework.

Animal-derived material built into a product can result in Red
classification.

Businesses that directly sell or serve animal food can move to a worse
band depending on the scale of that activity.
""",

        "limitations": """
LIMITATIONS

AIM states that ahimsa is a subjective idea and that no classification
can be fully objective.

Ratings rely on publicly available information and judgement and may
change when additional information becomes available.

The AQ band is a broad indication rather than an exact measurement.
""",
    }

    if any(word in question for word in ["finance", "bank", "investment", "capital"]):
        relevant = sections["finance"]
    elif any(word in question for word in ["red", "orange", "green", "band", "category", "aq"]):
        relevant = sections["bands"] + "\n" + sections["core_minority"]
    elif any(word in question for word in ["classify", "classification", "methodology", "how does aim"]):
        relevant = sections["process"]
    elif any(word in question for word in ["pharma", "pharmaceutical", "cosmetic", "pesticide", "insect", "rodent", "animal food"]):
        relevant = sections["special_rules"]
    elif any(word in question for word in ["limitation", "objective", "subjective"]):
        relevant = sections["limitations"]
    else:
        relevant = sections["bands"] + "\n" + sections["core_minority"]

    return relevant.strip()

def ask_local_ai(prompt):
    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False,
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result.get("response", "").strip()

    except Exception as exc:
        return f"AI explanation is currently unavailable: {exc}"

# Future historical-data interface.
# This remains empty until a real market-data source is added.
YAHOO_TICKER_MAP = {
    "Pranav Constructions Ltd": "PRANAV.NS",
}

BENCHMARK_TICKER = "^NSEI"

def load_historical_data(company_name):
    yahoo_ticker = YAHOO_TICKER_MAP.get(company_name)

    if not yahoo_ticker:
        return pd.DataFrame()

    company_data = yf.download(
        yahoo_ticker,
        period="1y",
        auto_adjust=False,
        progress=False,
    )

    benchmark_data = yf.download(
        BENCHMARK_TICKER,
        period="1y",
        auto_adjust=False,
        progress=False,
    )

    if company_data.empty or benchmark_data.empty:
        return pd.DataFrame()

    company_close = company_data["Close"]
    benchmark_close = benchmark_data["Close"]

    if isinstance(company_close, pd.DataFrame):
        company_close = company_close.iloc[:, 0]

    if isinstance(benchmark_close, pd.DataFrame):
        benchmark_close = benchmark_close.iloc[:, 0]

    historical = pd.concat(
        [
            company_close.rename("Price"),
            benchmark_close.rename("Benchmark"),
        ],
        axis=1,
        join="inner",
    )

    historical = historical.reset_index()
    historical["Company Name"] = company_name

    return historical[
        ["Company Name", "Date", "Price", "Benchmark"]
    ]

def load_data():
    if not DATA_FILE.exists():
        st.error("AQ dataset not found.")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
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


def normalise_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_matches(company_input):
    query = normalise_text(company_input)

    if not query:
        return pd.DataFrame()

    normalized_names = df["Company Name"].apply(normalise_text)

    exact_matches = df[normalized_names == query]

    if not exact_matches.empty:
        return exact_matches

    partial_mask = normalized_names.apply(
        lambda name: query in name or name in query
    )

    return df[partial_mask]

def search_dataset(question, max_results=8):
    """
    Route natural-language questions to the most relevant AIM dataset
    records.

    Retrieval only. This function does not classify companies or make
    investment recommendations.
    """

    query = normalise_text(question)

    if not query:
        return pd.DataFrame()
    
    methodology_only_terms = [
        "what does red mean",
        "what does orange mean",
        "what does green mean",
        "what does the green aq category mean",
        "what does the orange aq category mean",
        "what does the red aq category mean",
        "what is red",
        "what is orange",
        "what is green",
        "what does the red aq band mean",
        "what does the orange aq band mean",
        "what does the green aq band mean",
        "what is the red aq band",
        "what is the orange aq band",
        "what is the green aq band",
    ]

    if any(term in query for term in methodology_only_terms):
        return pd.DataFrame()

    # ---------------------------------------------------------
    # 1. Methodology questions
    # ---------------------------------------------------------

    methodology_terms = [
        "what is aq",
        "what does aq mean",
        "what is ahimsa quotient",
        "what does green mean",
        "what does orange mean",
        "what does red mean",
        "what is green",
        "what is orange",
        "what is red",
        "how does aim classify",
        "how are companies classified",
        "classification process",
        "classification methodology",
        "aq methodology",
        "core vs minority",
        "core or minority",
        "finance companies",
        "special rules",
        "limitations",
        "himsak",
        "ahimsak",
    ]

    if any(term in query for term in methodology_terms):
        return pd.DataFrame()

    # ---------------------------------------------------------
    # 2. Category questions
    # ---------------------------------------------------------

        
    # ---------------------------------------------------------
    # 3. Company-specific questions
    # ---------------------------------------------------------

    company_names = df["Company Name"].apply(normalise_text)

    phrase_matches = []

    for index, company_name in company_names.items():
        company_words = company_name.split()

        if len(company_words) >= 2:
            for phrase_length in range(
                min(4, len(company_words)),
                1,
                -1,
            ):
                phrase = " ".join(company_words[:phrase_length])

                if len(phrase) >= 5 and phrase in query:
                    phrase_matches.append(index)
                    break

    if phrase_matches:
        return df.loc[phrase_matches].head(max_results)

    # ---------------------------------------------------------
    # 4. Individual company-name word matching
    # ---------------------------------------------------------

    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "how", "i", "in", "is", "it", "me", "of", "on",
        "or", "the", "this", "to", "what", "why", "with",
        "tell", "about", "does", "do", "can", "company",
    }

    query_words = [
        word
        for word in query.split()
        if word not in stop_words and len(word) >= 3
    ]

    if not query_words:
        return pd.DataFrame()

    company_matches = []

    for index, company_name in company_names.items():
        company_words = set(company_name.split())

        overlap = sum(
            word in company_words
            for word in query_words
        )

        if overlap >= 1:
            company_matches.append((index, overlap))

    if company_matches:
        company_matches.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        indexes = [
            item[0]
            for item in company_matches[:max_results]
        ]

        return df.loc[indexes]

    # ---------------------------------------------------------
    # 5. Broader dataset search
    # ---------------------------------------------------------

    searchable = (
        df["Company Name"].fillna("")
        + " "
        + df["Symbol"].fillna("")
        + " "
        + df["Macro Sector"].fillna("")
        + " "
        + df["Sector"].fillna("")
        + " "
        + df["Industry"].fillna("")
        + " "
        + df["Basic Industry"].fillna("")
        + " "
        + df["Category"].fillna("")
        + " "
        + df["Justification"].fillna("")
    ).apply(normalise_text)

    scores = searchable.apply(
        lambda text: sum(
            1 for word in query_words
            if word in text
        )
    )

    results = df.loc[scores > 0].copy()

    if results.empty:
        return pd.DataFrame()

    results["_search_score"] = scores[scores > 0]

    return (
        results
        .sort_values("_search_score", ascending=False)
        .head(max_results)
        .drop(columns=["_search_score"])
    )

def display_company_result(row):
    st.divider()
    st.subheader(row["Company Name"])

    band = row["Category"].strip().title()

    band_descriptions = {
        "Green": (
            "Lower animal-harm exposure according to the current "
            "AIM AQ dataset."
        ),
        "Orange": (
            "Intermediate animal-harm exposure according to the "
            "current AIM AQ dataset."
        ),
        "Red": (
            "Higher animal-harm exposure according to the current "
            "AIM AQ dataset."
        ),
    }

    band_colors = {
        "Green": "#2E7D32",
        "Orange": "#EF6C00",
        "Red": "#C62828",
    }

    color = band_colors.get(band, "#666666")

    description = band_descriptions.get(
        band,
        "AQ classification according to the current AIM AQ dataset.",
    )

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            padding: 8px 18px;
            border-radius: 20px;
            background-color: {color};
            color: white;
            font-size: 20px;
            font-weight: 600;
            margin: 8px 0 12px 0;
        ">
            AQ Band: {band}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(description)

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

    st.markdown("#### Ask about this company")

    question = st.text_input(
        "Ask a question",
        placeholder="Ask anything about this company's AQ classification...",
        key=f"ai_question_{row['Company Name']}",
    )

    query = normalise_text(question)

    if st.button(
        "Ask",
        key=f"ai_explain_{row['Company Name']}",
    ):
        if not question.strip():
            st.warning("Please enter a question.")
            return
        
        context = f"""
Company: {row["Company Name"]}
AQ Band: {band}
AIM Justification: {row["Justification"]}

Company information:
Symbol: {row["Symbol"]}
Macro Sector: {row["Macro Sector"]}
Sector: {row["Sector"]}
Industry: {row["Industry"]}
Basic Industry: {row["Basic Industry"]}
"""

        prompt = build_ai_prompt(
            question,
            context,
        )

        with st.spinner("Generating explanation..."):
            answer = ask_local_ai(prompt)

        st.markdown("**AI explanation**")
        st.write(answer)

        st.caption(
            "AQ classification remains determined by the AIM AQ dataset."
        ) 

    st.markdown("#### Historical performance")

    historical = load_historical_data(row["Company Name"])

    if historical.empty:
        st.info(
            "Historical market data is not currently available "
            "for this company in the prototype."
        )
    else:
        company_start = historical["Price"].iloc[0]
        company_end = historical["Price"].iloc[-1]

        benchmark_start = historical["Benchmark"].iloc[0]
        benchmark_end = historical["Benchmark"].iloc[-1]

        company_return = (
            (company_end / company_start) - 1
        ) * 100

        benchmark_return = (
            (benchmark_end / benchmark_start) - 1
        ) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Company — 1 year",
                f"{company_return:+.1f}%",
            )

        with col2:
            st.metric(
                "NIFTY 50 — same period",
                f"{benchmark_return:+.1f}%",
            )

        chart_data = historical[["Price", "Benchmark"]].copy()

        chart_data = (
            chart_data
            / chart_data.iloc[0]
            * 100
        )

        chart_data.columns = [
            row["Company Name"],
            "NIFTY 50",
        ]

        st.line_chart(chart_data)

        st.caption(
            "Historical performance over the available 1-year period. "
            "Historical performance does not indicate future performance."
        )

def display_not_assessed():
    st.warning(
        "We couldn't find an AQ assessment for this company "
        "in the current dataset."
    )

    st.write(
        "This does not mean that the company has a particular "
        "AQ classification."
    )
def chat_page():
    st.title("Ask AIM")

    st.write(
        "Ask questions about AIM's Ahimsa Quotient, company "
        "classifications, and the AIM methodology."
    )

    question = st.text_input(
        "Your question",
        placeholder="Ask anything about ahimsak investment...",
        key="general_chat_question",
    )

    if st.button(
        "Ask",
        key="general_chat_ask",
    ):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        query = normalise_text(question)

        category_requested = None

        for possible_category in ["green", "orange", "red"]:
            if possible_category in query:
                category_requested = possible_category.title()
                break

        if (
            category_requested
            and "how many" in query
            and "compan" in query
        ):
            count = (
                df["Category"]
                .str.lower()
                .eq(category_requested.lower())
                .sum()
            )

            st.markdown("**AI answer**")
            st.write(
                f"There are {count} companies classified as "
                f"{category_requested} in the supplied AIM AQ dataset."
            )
            return

        results = search_dataset(question)

        context = ""

        if not results.empty:
            context = results.to_string(index=False)

        prompt = build_ai_prompt(
            question,
            context,
        )

        with st.spinner("Searching AIM information..."):
            answer = ask_local_ai(prompt)

        st.markdown("**AI answer**")
        st.write(answer)


def home_page():
    st.title("AIM Ahimsa Quotient")

    st.write(
        "Check companies against AIM's Ahimsa Quotient "
        "and explore their AQ classification."
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

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

    with col3:
        if st.button(
            "Build a portfolio",
            use_container_width=True,
        ):
            st.session_state["page"] = "builder"
            st.rerun()
   
    with col4:
        if st.button(
            "Ask AIM",
            use_container_width=True,
        ):
            st.session_state["page"] = "chat"
            st.rerun()

    if st.button(
        "Upload a portfolio",
        use_container_width=True,
    ):
        st.session_state["page"] = "upload"
        st.rerun()
    
    st.write("")

    with st.expander("What is AQ?"):
        st.write(
            "The Ahimsa Quotient (AQ) is AIM's framework for "
            "classifying companies according to their relationship "
            "with animal harm, based on the AIM AQ dataset used "
            "for this prototype."
        )


def company_page():
    if st.button("← Back to home"):
        st.session_state["page"] = "home"
        st.session_state.pop("selected_company", None)
        st.rerun()

    st.title("Check a company")

    company_options = [""] + sorted(
        df["Company Name"].dropna().unique().tolist()
    )

    company_input = st.selectbox(
        "Company name",
        company_options,
        index=0,
        placeholder="Start typing a company name",
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
            st.session_state["selected_company"] = (
                matches.iloc[0]["Company Name"]
            )
            st.session_state["company_input"] = company_input

        elif len(matches) > 1:
            st.session_state["ambiguous_matches"] = (
                matches["Company Name"].tolist()
            )
            st.session_state["company_input"] = company_input

        else:
            st.session_state.pop("selected_company", None)
            st.session_state.pop("ambiguous_matches", None)
            st.session_state["company_input"] = company_input

    if "ambiguous_matches" in st.session_state:
        options = st.session_state["ambiguous_matches"]

        st.info("We found multiple possible matches.")

        selected_company = st.selectbox(
            "Select the company you mean",
            options,
        )

        st.session_state["selected_company"] = selected_company

    if "selected_company" in st.session_state:
        selected_company = st.session_state["selected_company"]

        selected_matches = find_matches(selected_company)

        if len(selected_matches) > 0:
            display_company_result(selected_matches.iloc[0])

def analyse_portfolio_inputs(companies):
    """
    Deterministically match each portfolio input against the AQ dataset.
    """

    results = []

    for company in companies:
        matches = find_matches(company)

        if len(matches) == 1:
            row = matches.iloc[0]

            results.append({
                "User input": company,
                "Matched company": row["Company Name"],
                "AQ Band": row["Category"].title(),
                "Status": "Assessed",
            })

        elif len(matches) > 1:
            results.append({
                "User input": company,
                "Matched company": "",
                "AQ Band": "",
                "Status": "Ambiguous / needs selection",
            })

        else:
            results.append({
                "User input": company,
                "Matched company": "",
                "AQ Band": "",
                "Status": "Not currently assessed",
            })

    return pd.DataFrame(results)



def portfolio_upload_page():
    if st.button("← Back to home", key="upload_back_home"):
        st.session_state["page"] = "home"
        st.rerun()

    st.title("Upload and assess a portfolio")
    st.write(
        "Upload a CSV or Excel portfolio export. "
        "AQ classifications come only from the current AIM AQ dataset."
    )

    uploaded_file = st.file_uploader(
        "Choose a portfolio file",
        type=["csv", "xlsx", "xls"],
        key="portfolio_upload_file",
    )

    if uploaded_file is None:
        st.info("Upload a CSV or Excel file to begin.")
        return

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            uploaded_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        else:
            uploaded_df = pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        return

    if uploaded_df.empty or len(uploaded_df.columns) == 0:
        st.warning("The uploaded file contains no usable rows or columns.")
        return

    st.write(f"Rows found: {len(uploaded_df)}")

    columns = list(uploaded_df.columns)
    optional_columns = ["(Not available)"] + columns

    st.subheader("Map your file columns")
    name_col = st.selectbox(
        "Company name column (required)",
        columns,
        key="upload_name_col",
    )
    isin_col = st.selectbox(
        "ISIN column (optional)",
        optional_columns,
        key="upload_isin_col",
    )
    symbol_col = st.selectbox(
        "Stock symbol column (optional)",
        optional_columns,
        key="upload_symbol_col",
    )
    value_col = st.selectbox(
        "Market value column (optional)",
        optional_columns,
        key="upload_value_col",
    )

    if not st.button(
        "Match holdings to AIM AQ",
        type="primary",
        key="run_portfolio_upload",
    ):
        return

    def clean_value(value):
        if pd.isna(value):
            return ""
        return str(value).strip()

    aq_data = df.copy()
    aq_data["_name_key"] = aq_data["Company Name"].apply(normalise_text)
    aq_data["_isin_key"] = (
        aq_data["ISIN Code"].fillna("").astype(str).str.strip().str.upper()
    )
    aq_data["_symbol_key"] = (
        aq_data["Symbol"].fillna("").astype(str).str.strip().str.upper()
    )

    results = []

    for _, holding in uploaded_df.iterrows():
        original_name = clean_value(holding[name_col])
        isin = (
            clean_value(holding[isin_col]).upper()
            if isin_col != "(Not available)"
            else ""
        )
        symbol = (
            clean_value(holding[symbol_col]).upper()
            if symbol_col != "(Not available)"
            else ""
        )

        market_value = None
        if value_col != "(Not available)":
            raw_value = clean_value(holding[value_col])
            if raw_value:
                market_value = pd.to_numeric(
                    raw_value.replace(",", "").replace("₹", "").strip(),
                    errors="coerce",
                )

        matched_company = ""
        aq_band = ""
        status = "Not currently assessed"
        candidates = pd.DataFrame()

        if not original_name and not isin and not symbol:
            status = "Missing company identifiers"

        elif isin:
            candidates = aq_data[aq_data["_isin_key"] == isin]
            if len(candidates) == 1:
                status = "Assessed — ISIN match"
            elif len(candidates) > 1:
                status = "Ambiguous ISIN — needs review"
            else:
                status = "ISIN not found — needs review"
                candidates = pd.DataFrame()

        elif symbol:
            candidates = aq_data[aq_data["_symbol_key"] == symbol]
            if len(candidates) == 1:
                status = "Assessed — symbol match"
            elif len(candidates) > 1:
                status = "Ambiguous symbol — needs review"
            else:
                status = "Symbol not found — needs review"
                candidates = pd.DataFrame()

        elif original_name:
            name_key = normalise_text(original_name)
            candidates = aq_data[aq_data["_name_key"] == name_key]
            if len(candidates) == 1:
                status = "Assessed — exact name match"
            elif len(candidates) > 1:
                status = "Ambiguous name — needs review"
            else:
                status = "Not currently assessed"

        if len(candidates) == 1 and status.startswith("Assessed"):
            matched = candidates.iloc[0]
            matched_company = matched["Company Name"]
            aq_band = str(matched["Category"]).strip().title()

        results.append({
            "Original holding": original_name,
            "Uploaded ISIN": isin,
            "Uploaded symbol": symbol,
            "Matched company": matched_company,
            "AQ Band": aq_band,
            "Match status": status,
            "Market Value": market_value,
        })

    result_df = pd.DataFrame(results)

    st.subheader("Portfolio matching results")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    assessed_count = int(
        result_df["Match status"].str.startswith("Assessed").sum()
    )
    st.metric("Holdings matched to AQ", f"{assessed_count} / {len(result_df)}")

    st.caption(
        "Only exact identifier or exact company-name matches are assessed. "
        "Unmatched and ambiguous holdings are not assigned an AQ category. "
        "Market values are displayed as supplied; exposure calculations "
        "will be added after this matching workflow is tested."
    )

def portfolio_page():
    if st.button("← Back to home"):
        st.session_state["page"] = "home"
        st.rerun()

    st.title("Assess a portfolio")

    st.write("Enter one company name per line.")

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

        results_df = analyse_portfolio_inputs(companies)

        st.subheader("Portfolio results")

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
        )

        assessed_results = results_df[
            results_df["Status"] == "Assessed"
        ]

        green_count = (
            assessed_results["AQ Band"]
            .eq("Green")
            .sum()
        )

        orange_count = (
            assessed_results["AQ Band"]
            .eq("Orange")
            .sum()
        )

        red_count = (
            assessed_results["AQ Band"]
            .eq("Red")
            .sum()
        )

        not_assessed_count = (
            results_df["Status"]
            .eq("Not currently assessed")
            .sum()
        )

        ambiguous_count = (
            results_df["Status"]
            .eq("Ambiguous / needs selection")
            .sum()
        )

        st.subheader("Holding counts")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Green", green_count)
        col2.metric("Orange", orange_count)
        col3.metric("Red", red_count)
        col4.metric("Not assessed", not_assessed_count)

        if ambiguous_count:
            st.info(
                f"{ambiguous_count} holding(s) have multiple possible "
                "matches and need selection."
            )

        total_holdings = len(results_df)
        assessed_count = len(assessed_results)

        st.subheader("Portfolio overview")

        st.write(
            f"**AQ coverage:** {assessed_count} of {total_holdings} "
            "holdings have an AIM AQ classification in the current dataset."
        )

        st.markdown("**AQ distribution**")

        distribution_df = pd.DataFrame({
            "AQ Band": ["Green", "Orange", "Red"],
            "Holdings": [
                green_count,
                orange_count,
                red_count,
            ],
        })

        st.dataframe(
            distribution_df,
            use_container_width=True,
            hide_index=True,
        )

        unassessed = results_df[
            results_df["Status"] == "Not currently assessed"
        ]

        if not unassessed.empty:
            st.markdown("**Not currently assessed**")

            for company in unassessed["User input"]:
                st.write(f"• {company}")

        st.subheader("Portfolio observations")

        observations = [
            f"Your portfolio contains {assessed_count} assessed holdings."
        ]

        if green_count:
            observations.append(
                f"{green_count} holding(s) are Green."
            )

        if orange_count:
            observations.append(
                f"{orange_count} holding(s) are Orange."
            )

        if red_count:
            observations.append(
                f"{red_count} holding(s) are Red."
            )

        if not_assessed_count:
            observations.append(
                f"{not_assessed_count} holding(s) are not currently "
                "assessed in the prototype dataset."
            )

        assessed_sectors = assessed_results.merge(
            df[["Company Name", "Sector"]],
            left_on="Matched company",
            right_on="Company Name",
            how="left",
        )

        sector_count = (
            assessed_sectors["Sector"]
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )

        if sector_count:
            observations.append(
                f"Your assessed holdings span {sector_count} sectors."
            )

        for observation in observations:
            st.write(f"• {observation}")

        if not assessed_results.empty:
            st.subheader("View company details")

            selected_company = st.selectbox(
                "Select a matched company",
                assessed_results["Matched company"].tolist(),
            )

            selected_row = df[
                df["Company Name"] == selected_company
            ].iloc[0]

            display_company_result(selected_row)


def calculate_target_holding_counts(selected_bands, number_of_holdings, allocations):
    """
    Convert percentage allocations into whole-number holding counts.

    Uses the largest-remainder method so the final counts sum exactly
    to the requested number of holdings.
    """

    exact_counts = {
        band: number_of_holdings * allocations[band] / 100
        for band in selected_bands
    }

    base_counts = {
        band: int(exact_counts[band] // 1)
        for band in selected_bands
    }

    remaining = number_of_holdings - sum(base_counts.values())

    remainders = sorted(
        selected_bands,
        key=lambda band: (
            exact_counts[band] - base_counts[band],
            -selected_bands.index(band),
        ),
        reverse=True,
    )

    for band in remainders[:remaining]:
        base_counts[band] += 1

    return base_counts


def get_eligible_companies(selected_bands):
    """
    Return companies whose authoritative dataset Category matches
    one of the selected AQ bands.
    """

    eligible = df[
        df["Category"].str.title().isin(selected_bands)
    ].copy()

    # Deterministic ordering.
    eligible = eligible.sort_values(
        by="Company Name"
    ).reset_index(drop=True)

    return eligible


def construct_illustrative_portfolio(
    selected_bands,
    number_of_holdings,
    allocations,
):
    """
    Construct a deterministic illustrative portfolio from the
    eligible AQ dataset.
    """

    target_counts = calculate_target_holding_counts(
        selected_bands,
        number_of_holdings,
        allocations,
    )

    eligible = get_eligible_companies(selected_bands)

    available_counts = (
        eligible["Category"]
        .str.title()
        .value_counts()
        .to_dict()
    )

    insufficient_bands = []

    for band in selected_bands:
        required = target_counts.get(band, 0)
        available = available_counts.get(band, 0)

        if required > available:
            insufficient_bands.append({
                "band": band,
                "required": required,
                "available": available,
            })

    if insufficient_bands:
        return None, target_counts, available_counts, insufficient_bands

    selected_rows = []

    for band in selected_bands:
        required = target_counts[band]

        band_companies = eligible[
            eligible["Category"].str.title() == band
        ].head(required)

        for _, row in band_companies.iterrows():
            selected_rows.append(row)

    portfolio_df = pd.DataFrame(selected_rows)

    # Preserve deterministic band ordering followed by company name.
    portfolio_df["AQ Band"] = portfolio_df["Category"].str.title()

    band_order = {
        band: index
        for index, band in enumerate(selected_bands)
    }

    portfolio_df["_band_order"] = portfolio_df["AQ Band"].map(
        band_order
    )

    portfolio_df = portfolio_df.sort_values(
        by=["_band_order", "Company Name"]
    ).reset_index(drop=True)

    portfolio_df = portfolio_df.drop(columns=["_band_order"])

    # Equal allocation within each AQ band.
    allocations_list = []

    for _, row in portfolio_df.iterrows():
        band = row["AQ Band"]
        count_in_band = target_counts[band]

        allocation = allocations[band] / count_in_band

        allocations_list.append(allocation)

    portfolio_df["Illustrative Allocation"] = allocations_list

    return (
        portfolio_df,
        target_counts,
        available_counts,
        [],
    )


def portfolio_builder_page():
    if st.button("← Back to home"):
        st.session_state["page"] = "home"
        st.rerun()

    st.title("Build a portfolio")

    st.write(
        "Construct an illustrative portfolio using your selected "
        "AQ-band criteria and the companies currently available "
        "in the AIM AQ prototype dataset."
    )

    st.info(
        "This is a rule-based illustrative tool. It does not provide "
        "personalised investment advice or determine whether a company "
        "is a suitable investment."
    )

    st.subheader("1. AQ bands to include")

    available_bands = ["Green", "Orange", "Red"]

    selected_bands = []

    for band in available_bands:
        if st.checkbox(band, key=f"builder_band_{band}"):
            selected_bands.append(band)

    if not selected_bands:
        st.warning("Select at least one AQ band.")
        return

    st.subheader("2. Number of holdings")

    number_of_holdings = st.number_input(
        "Number of holdings",
        min_value=1,
        max_value=len(df),
        value=min(5, len(df)),
        step=1,
    )

    st.subheader("3. Desired allocation by AQ band")

    allocations = {}

    if len(selected_bands) == 1:
        allocations[selected_bands[0]] = 100.0

        st.write(
            f"**{selected_bands[0]} allocation: 100%**"
        )

    else:
        remaining = 100.0

        for index, band in enumerate(selected_bands):
            if index == len(selected_bands) - 1:
                default_value = max(0.0, remaining)
            else:
                default_value = 100.0 / len(selected_bands)

            allocations[band] = st.number_input(
                f"{band} allocation (%)",
                min_value=0.0,
                max_value=100.0,
                value=round(default_value, 1),
                step=1.0,
                key=f"builder_allocation_{band}",
            )

            remaining -= allocations[band]

        allocation_total = sum(allocations.values())

        st.write(
            f"**Allocation total: {allocation_total:.1f}%**"
        )

        if abs(allocation_total - 100.0) > 0.01:
            st.error(
                "Your allocations must sum to 100%."
            )
            return

        zero_allocations = [
            band
            for band in selected_bands
            if allocations[band] <= 0
        ]

        if zero_allocations:
            st.error(
                "Each selected AQ band must have an allocation greater "
                "than 0%."
            )
            return

    st.subheader("Available AQ universe")

    eligible_companies = get_eligible_companies(selected_bands)

    universe_counts = (
        eligible_companies["Category"]
        .str.title()
        .value_counts()
        .reindex(selected_bands, fill_value=0)
    )

    universe_df = pd.DataFrame({
        "AQ Band": selected_bands,
        "Eligible companies": [
            int(universe_counts[band])
            for band in selected_bands
        ],
    })

    st.dataframe(
        universe_df,
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "Construct illustrative portfolio",
        type="primary",
        use_container_width=True,
    ):
        (
            portfolio_df,
            target_counts,
            available_counts,
            insufficient_bands,
        ) = construct_illustrative_portfolio(
            selected_bands,
            number_of_holdings,
            allocations,
        )

        if insufficient_bands:
            st.error(
                "These criteria cannot currently be satisfied using "
                "the available AQ dataset."
            )

            st.write(
                "The prototype will not fabricate additional companies "
                "to satisfy the requested criteria."
            )

            st.markdown("**Constraint details**")

            for item in insufficient_bands:
                st.write(
                    f"• {item['band']}: {item['required']} holding(s) "
                    f"required, but only {item['available']} eligible "
                    "companies are present in the "
                    "current prototype dataset."
                )

            st.markdown("**Eligible universe counts**")

            constraint_df = pd.DataFrame({
                "AQ Band": selected_bands,
                "Eligible companies": [
                    available_counts.get(band, 0)
                    for band in selected_bands
                ],
                "Required holdings": [
                    target_counts.get(band, 0)
                    for band in selected_bands
                ],
            })

            st.dataframe(
                constraint_df,
                use_container_width=True,
                hide_index=True,
            )

            return

        st.subheader("Your criteria")

        st.write(
            "**AQ bands:** "
            + " + ".join(selected_bands)
        )

        st.write(
            f"**Number of holdings:** {number_of_holdings}"
        )

        target_text = " | ".join(
            f"{band} {allocations[band]:.1f}%"
            for band in selected_bands
        )

        st.write(
            f"**Target allocation:** {target_text}"
        )

        st.subheader("Illustrative portfolio")

        display_df = portfolio_df[
            [
                "Company Name",
                "AQ Band",
                "Illustrative Allocation",
            ]
        ].copy()

        display_df["Illustrative Allocation"] = (
            display_df["Illustrative Allocation"]
            .map(lambda value: f"{value:.1f}%")
        )

        display_df = display_df.rename(
            columns={
                "Company Name": "Company",
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        total_allocation = portfolio_df[
            "Illustrative Allocation"
        ].sum()

        st.write(
            f"**Total illustrative allocation: "
            f"{total_allocation:.1f}%**"
        )

        st.subheader("Why each company was included")

        for _, row in portfolio_df.iterrows():
            st.write(
                f"**{row['Company Name']}** — Included because it is "
                f"classified {row['AQ Band']} in the AIM AQ dataset "
                "and was selected to satisfy your specified "
                "portfolio criteria."
            )

        st.subheader("How this portfolio was constructed")

        st.write(
            "1. The application used your selected AQ bands."
        )

        st.write(
            "2. It used your selected number of holdings."
        )

        st.write(
            "3. It used your specified allocation targets."
        )

        st.write(
            "4. It identified eligible companies in the current "
            "AIM AQ dataset."
        )

        st.write(
            "5. It converted the target percentages into whole-number "
            "holding counts and selected companies deterministically "
            "from the eligible universe."
        )

        st.caption(
            "This output is an illustrative portfolio constructed "
            "from your selected criteria. It is not a recommendation "
            "or personalised investment advice."
        )


if "page" not in st.session_state:
    st.session_state["page"] = "home"


if st.session_state["page"] == "home":
    home_page()

elif st.session_state["page"] == "company":
    company_page()

elif st.session_state["page"] == "portfolio":
    portfolio_page()

elif st.session_state["page"] == "builder":
    portfolio_builder_page()
elif st.session_state["page"] == "chat":
    chat_page()    

elif st.session_state["page"] == "upload":
    portfolio_upload_page()
