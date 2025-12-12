import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Child Support Fiscal Leadership Demo",
    layout="wide"
)

# --------------------------------------------------
# Data loader (Cloud-safe)
# --------------------------------------------------
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    files = {
        "complexity": DATA_DIR / "county_complexity_scores.csv",
        "budget": DATA_DIR / "budget.csv",
        "monthly": DATA_DIR / "monthly_finance.csv",
        "quarterly": DATA_DIR / "quarterly_ffp.csv",
        "customer": DATA_DIR / "customer_payment_breakdown.csv",
    }

    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        st.error("Required demo data files are missing.")
        st.markdown("**Expected files:**")
        for k, p in files.items():
            st.code(str(p))
        st.info(
            "Fix: Ensure the `/data` folder exists in the GitHub repo "
            "and includes all CSV files (committed and pushed)."
        )
        st.stop()

    complexity = pd.read_csv(files["complexity"])
    budget = pd.read_csv(files["budget"])
    monthly = pd.read_csv(files["monthly"], parse_dates=["month"])
    quarterly = pd.read_csv(files["quarterly"])
    customer = pd.read_csv(files["customer"])

    return complexity, budget, monthly, quarterly, customer


complexity, budget, monthly, quarterly, customer = load_data()

# --------------------------------------------------
# Metric functions
# --------------------------------------------------
def compute_cci(df):
    weights = {
        "dv_score": 0.25,
        "homelessness_score": 0.20,
        "unemployment_score": 0.15,
        "rural_access_score": 0.15,
        "court_delay_score": 0.15,
        "ncp_volatility_score": 0.10,
    }
    out = df.copy()
    out["county_complexity_index"] = sum(out[k] * v for k, v in weights.items())
    return out, weights


def rag_band(value, green_max=0.90, amber_max=1.10):
    if pd.isna(value):
        return "N/A"
    if value < green_max:
        return "Green"
    if value <= amber_max:
        return "Amber"
    return "Red"


# --------------------------------------------------
# Pre-compute metrics
# --------------------------------------------------
complexity_cci, weights = compute_cci(complexity)

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.title("Leadership Controls")

county = st.sidebar.selectbox(
    "Select county",
    sorted(complexity_cci["county_name"].unique())
)

show_statewide = st.sidebar.checkbox(
    "Show statewide comparison",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("Scenario Sliders (Demo)")

unemployment_shock = st.sidebar.slider(
    "Unemployment shock (±%)",
    -10, 20, 0
)

coding_quality = st.sidebar.slider(
    "IV-D coding quality (eligible % shift)",
    -10, 10, 0
)

# --------------------------------------------------
# Filter county
# --------------------------------------------------
c_row = complexity_cci[complexity_cci["county_name"] == county].iloc[0]
m_c = monthly[monthly["county_name"] == county].sort_values("month")
b_c = budget[budget["county_name"] == county]
q_c = quarterly[quarterly["county_name"] == county].copy()
cust_c = customer[customer["county_name"] == county]

# --------------------------------------------------
# Scenario effects (demo only)
# --------------------------------------------------
shock_factor = 1 - unemployment_shock / 100 * 0.6

m_c_s = m_c.copy()
m_c_s["collections_scn"] = m_c_s["collections"] * shock_factor
m_c_s["arrears_scn"] = (
    m_c_s["arrears_total"]
    + (m_c_s["collections"] - m_c_s["collections_scn"]).cumsum() * 0.25
)

q_c["eligible_pct_scn"] = np.clip(
    q_c["eligible_pct"] + coding_quality / 100,
    0.50, 0.98
)
q_c["eligible_expenses_scn"] = q_c["expenses"] * q_c["eligible_pct_scn"]
q_c["ffp_match_value_scn"] = q_c["eligible_expenses_scn"] * q_c["ffp_match_rate"]

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("Child Support Fiscal Metrics – Leadership Demo")
st.caption(
    "Unbranded demonstration using mock data to show how a holistic fiscal model "
    "supports equitable, informed decision-making."
)

k1, k2, k3, k4, k5 = st.columns(5)

annual_exp = m_c["expenses"].tail(12).sum()
cost_per_case = annual_exp / max(c_row["active_cases"], 1)

total_budget = b_c["allocated_amount"].sum()
budget_per_case = total_budget / max(c_row["active_cases"], 1)

fps = (cost_per_case * c_row["county_complexity_index"]) / max(budget_per_case, 1e-9)
fps_band = rag_band(fps)

pt_rate = m_c["pass_through_amount"].sum() / max(m_c["collections"].sum(), 1e-9)
ffp = q_c["ffp_match_value_scn"].tail(4).sum()

k1.metric("County Complexity Index", f"{c_row['county_complexity_index']:.2f}")
k2.metric("Cost per Case (12 mo)", f"${cost_per_case:,.0f}")
k3.metric("Budget per Case (FY)", f"${budget_per_case:,.0f}")
k4.metric("Fiscal Pressure Score", f"{fps:.2f}", fps_band)
k5.metric("FFP Match (last 4 qtrs)", f"${ffp:,.0f}")

st.markdown("---")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tabs = st.tabs([
    "Overview",
    "County Complexity",
    "Expense Mix",
    "Federal Match (FFP)",
    "Arrears Trends",
    "Customer Transparency",
    "Definitions"
])

# --------------------------------------------------
# Overview
# --------------------------------------------------
with tabs[0]:
    left, right = st.columns([2, 1])

    with left:
        st.line_chart(
            m_c_s[["month", "collections", "collections_scn", "expenses"]]
            .set_index("month"),
            height=300
        )

    with right:
        st.markdown("### Leadership signals")
        st.write(pd.DataFrame({
            "Metric": ["Fiscal Pressure", "Pass-through Rate", "Eligible % (scen.)"],
            "Value": [f"{fps:.2f}", f"{pt_rate*100:.1f}%", f"{q_c['eligible_pct_scn'].mean()*100:.1f}%"],
            "Status": [
                fps_band,
                "Green" if pt_rate >= 0.60 else "Amber" if pt_rate >= 0.55 else "Red",
                "Green" if q_c["eligible_pct_scn"].mean() >= 0.80 else "Amber"
            ]
        }))

# --------------------------------------------------
# County Complexity
# --------------------------------------------------
with tabs[1]:
    st.dataframe(
        complexity_cci[complexity_cci["county_name"] == county],
        use_container_width=True
    )

# --------------------------------------------------
# Expense Mix
# --------------------------------------------------
with tabs[2]:
    split = b_c.groupby("cost_center")["allocated_amount"].sum()
    st.bar_chart(split)

# --------------------------------------------------
# Federal Match
# --------------------------------------------------
with tabs[3]:
    q_c["quarter"] = q_c["quarter"].astype(str)
    st.line_chart(q_c.set_index("quarter")[["eligible_pct", "eligible_pct_scn"]])
    st.bar_chart(q_c.set_index("quarter")[["ffp_match_value", "ffp_match_value_scn"]])

# --------------------------------------------------
# Arrears
# --------------------------------------------------
with tabs[4]:
    st.line_chart(
        m_c_s[["month", "arrears_total", "arrears_scn"]].set_index("month")
    )

# --------------------------------------------------
# Customer Transparency
# --------------------------------------------------
with tabs[5]:
    avg = cust_c[["pass_through", "arrears_applied", "state_retained", "fees"]].mean()
    st.bar_chart(avg)

# --------------------------------------------------
# Definitions
# --------------------------------------------------
with tabs[6]:
    st.markdown("""
**County Complexity Index:** Weighted composite of structural risk drivers  
**Expense Mix Ratio:** Distribution of spending by function  
**Fiscal Pressure Score:** Equity-adjusted cost strain indicator  
**FFP Match:** Eligible expenses × 66%  
**Arrears Trajectory:** Direction and speed of arrears change
""")
