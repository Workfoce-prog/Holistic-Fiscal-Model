import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Child Support Fiscal Leadership Demo", layout="wide")

@st.cache_data
def load_data():
    complexity = pd.read_csv("data/county_complexity_scores.csv")
    budget = pd.read_csv("data/budget.csv")
    monthly = pd.read_csv("data/monthly_finance.csv", parse_dates=["month"])
    quarterly = pd.read_csv("data/quarterly_ffp.csv")
    customer = pd.read_csv("data/customer_payment_breakdown.csv")
    return complexity, budget, monthly, quarterly, customer

complexity, budget, monthly, quarterly, customer = load_data()

# ---- Metric functions (A–G subset for this demo) ----
def compute_cci(df):
    # Government-style weights (adjustable)
    w = {
        "dv_score": 0.25,
        "homelessness_score": 0.20,
        "unemployment_score": 0.15,
        "rural_access_score": 0.15,
        "court_delay_score": 0.15,
        "ncp_volatility_score": 0.10,
    }
    out = df.copy()
    out["county_complexity_index"] = sum(out[k]*v for k,v in w.items())
    return out, w

def rag_band(value, green_max=0.90, amber_max=1.10):
    # For Fiscal Pressure Score: lower is better
    if pd.isna(value):
        return "N/A"
    if value < green_max:
        return "Green"
    if value <= amber_max:
        return "Amber"
    return "Red"

# Precompute CCI
complexity_cci, weights = compute_cci(complexity)

# Sidebar controls
st.sidebar.title("Leadership Controls")
county = st.sidebar.selectbox("Select county", sorted(complexity_cci["county_name"].unique()))
show_statewide = st.sidebar.checkbox("Show statewide comparison", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Scenario Sliders (Demo)")
unemployment_shock = st.sidebar.slider("Unemployment shock (±%)", -10, 20, 0, help="Demo only: shifts collections and arrears forecasts.")
coding_quality = st.sidebar.slider("IV-D coding quality (eligible % shift)", -10, 10, 0, help="Demo only: changes eligible % for match value.")

# Filter county
c_row = complexity_cci[complexity_cci["county_name"] == county].iloc[0]
m_c = monthly[monthly["county_name"] == county].sort_values("month")
b_c = budget[budget["county_name"] == county]
q_c = quarterly[quarterly["county_name"] == county].copy()
cust_c = customer[customer["county_name"] == county]

# Apply scenario effects (simple demo transformations)
shock_factor = 1 - unemployment_shock/100.0 * 0.6  # unemployment reduces collections
m_c_s = m_c.copy()
m_c_s["collections_scn"] = m_c_s["collections"] * shock_factor
m_c_s["arrears_scn"] = m_c_s["arrears_total"] + (m_c_s["collections"] - m_c_s["collections_scn"]).cumsum() * 0.25

q_c["eligible_pct_scn"] = np.clip(q_c["eligible_pct"] + coding_quality/100.0, 0.50, 0.98)
q_c["eligible_expenses_scn"] = q_c["expenses"] * q_c["eligible_pct_scn"]
q_c["ffp_match_value_scn"] = q_c["eligible_expenses_scn"] * q_c["ffp_match_rate"]

# ---- Header KPIs ----
st.title("Child Support Fiscal Metrics – Leadership Demo (Unbranded)")
st.caption("Demonstration app with mock data to illustrate how a holistic fiscal model supports decisions across counties.")

k1, k2, k3, k4, k5 = st.columns(5)

# Cost per case (annualized from monthly)
annual_exp = m_c["expenses"].tail(12).sum()
cost_per_case = annual_exp / max(c_row["active_cases"], 1)

# Budget per case
total_budget = b_c["allocated_amount"].sum()
budget_per_case = total_budget / max(c_row["active_cases"], 1)

# Fiscal Pressure Score
fps = (cost_per_case * c_row["county_complexity_index"]) / max(budget_per_case, 1e-9)
fps_band = rag_band(fps)

# Pass-through rate
pt_rate = m_c["pass_through_amount"].sum() / max(m_c["collections"].sum(), 1e-9)

# Match value (last 4 quarters)
ffp = q_c["ffp_match_value_scn"].tail(4).sum()

k1.metric("County Complexity Index (CCI)", f"{c_row['county_complexity_index']:.2f}")
k2.metric("Cost per Case (12 mo)", f"${cost_per_case:,.0f}")
k3.metric("Budget per Case (FY)", f"${budget_per_case:,.0f}")
k4.metric("Fiscal Pressure Score (FPS)", f"{fps:.2f}", fps_band)
k5.metric("FFP Match (last 4 qtrs, scen.)", f"${ffp:,.0f}")

st.markdown("---")

# ---- Tabs ----
tabs = st.tabs([
    "Overview",
    "County Complexity",
    "Expense Mix",
    "Federal Match (FFP)",
    "Arrears Trends & Forecast",
    "Customer Transparency",
    "Data & Definitions"
])

with tabs[0]:
    st.subheader("Executive Overview")
    left, right = st.columns([2, 1])

    with left:
        chart_df = m_c_s[["month", "collections", "collections_scn", "expenses"]].set_index("month")
        st.line_chart(chart_df, height=300)

        st.markdown("**Interpretation (demo):**")
        st.markdown(
            "- Compare **Collections** vs **Expenses** to assess fiscal efficiency.\n"
            "- Use the scenario slider to see how an unemployment shock can reduce collections and increase arrears pressure."
        )

    with right:
        st.markdown("### RAG Summary")
        st.write(pd.DataFrame({
            "Metric": ["Fiscal Pressure Score", "Pass-through Rate", "Eligible % (avg, scen.)"],
            "Value": [f"{fps:.2f}", f"{pt_rate*100:.1f}%", f"{q_c['eligible_pct_scn'].mean()*100:.1f}%"],
            "Status": [fps_band,
                       "Green" if pt_rate >= 0.60 else "Amber" if pt_rate >= 0.55 else "Red",
                       "Green" if q_c['eligible_pct_scn'].mean() >= 0.80 else "Amber" if q_c['eligible_pct_scn'].mean() >= 0.72 else "Red"]
        }))

        st.markdown("### Leadership Questions this answers")
        st.markdown(
            "- Which counties show **fiscal strain** after adjusting for complexity?\n"
            "- Where do we have **coding/claim opportunities** to increase federal match?\n"
            "- Where are arrears trending up and what scenario risks drive it?"
        )

    if show_statewide:
        st.markdown("### County Comparison (Statewide)")
        comp = complexity_cci.copy()
        # compute CPC and FPS for each county for comparison
        rows = []
        for cname in comp["county_name"]:
            mc = monthly[monthly["county_name"] == cname].sort_values("month")
            bc = budget[budget["county_name"] == cname]
            crow = comp[comp["county_name"] == cname].iloc[0]
            annual_exp = mc["expenses"].tail(12).sum()
            cpc = annual_exp / max(crow["active_cases"], 1)
            bpc = bc["allocated_amount"].sum() / max(crow["active_cases"], 1)
            fps_i = (cpc * crow["county_complexity_index"]) / max(bpc, 1e-9)
            rows.append([cname, crow["county_complexity_index"], cpc, bpc, fps_i, rag_band(fps_i)])
        comp_tbl = pd.DataFrame(rows, columns=["County", "CCI", "Cost per Case", "Budget per Case", "FPS", "RAG"])
        st.dataframe(comp_tbl.sort_values("FPS", ascending=False), use_container_width=True)

with tabs[1]:
    st.subheader("County Complexity (CCI + component drivers)")

    st.markdown("**CCI weights (editable policy choice):**")
    w_df = pd.DataFrame({"Component": list(weights.keys()), "Weight": list(weights.values())})
    st.dataframe(w_df, use_container_width=True)

    comp_cols = ["dv_score","homelessness_score","unemployment_score","rural_access_score","court_delay_score","ncp_volatility_score"]
    comp_view = complexity_cci[complexity_cci["county_name"] == county][["county_name","fiscal_year"] + comp_cols + ["county_complexity_index"]]
    st.markdown("**Selected county profile:**")
    st.dataframe(comp_view, use_container_width=True)

    st.markdown("### Component comparison vs statewide")
    # Build comparison bars (county vs average=1.0)
    comp_compare = pd.DataFrame({
        "Component": ["DV", "Homelessness", "Unemployment", "Rural Access", "Court Delay", "NCP Volatility"],
        "County": [c_row["dv_score"], c_row["homelessness_score"], c_row["unemployment_score"], c_row["rural_access_score"], c_row["court_delay_score"], c_row["ncp_volatility_score"]],
        "State Avg": [1,1,1,1,1,1]
    }).set_index("Component")
    st.bar_chart(comp_compare, height=320)

    st.info("Use this view to explain *why* a county is complex (drivers), not just that it is complex.")

with tabs[2]:
    st.subheader("Expense Mix Ratio (where the dollars go)")

    # Annualized expense mix from last 12 months, allocate to cost centers by the FY budget split as a proxy
    total_exp_12 = m_c["expenses"].tail(12).sum()
    split = b_c.groupby("cost_center", as_index=False)["allocated_amount"].sum()
    split["share"] = split["allocated_amount"] / split["allocated_amount"].sum()
    split["expense_estimate_12mo"] = split["share"] * total_exp_12
    split_view = split[["cost_center","expense_estimate_12mo","share"]].sort_values("share", ascending=False).copy()
    split_view["share_pct"] = (split_view["share"]*100).round(1).astype(str) + "%"

    left, right = st.columns([1,1])
    with left:
        st.dataframe(split_view[["cost_center","expense_estimate_12mo","share_pct"]], use_container_width=True)
    with right:
        st.bar_chart(split_view.set_index("cost_center")[["expense_estimate_12mo"]], height=320)

    st.markdown("**Leadership interpretation:**")
    st.markdown(
        "- High **enforcement share** can indicate heavy arrears/noncompliance workload.\n"
        "- Higher **technology share** may reflect modernization investments (often good if tied to ROI).\n"
        "- Mix helps identify outliers and guide operational support."
    )

with tabs[3]:
    st.subheader("Federal Match (FFP) – eligible %, match value, and scenario")

    view = q_c.copy()
    view["quarter"] = view["quarter"].astype(str)

    left, right = st.columns([2,1])
    with left:
        st.line_chart(view.set_index("quarter")[["eligible_pct", "eligible_pct_scn"]], height=280)
        st.caption("Eligible % baseline vs scenario (coding quality slider).")
        st.bar_chart(view.set_index("quarter")[["ffp_match_value", "ffp_match_value_scn"]], height=280)
        st.caption("FFP match value baseline vs scenario.")

    with right:
        st.markdown("### Example calculation")
        st.code("FFP Match Value = Eligible Expenses × 0.66", language="text")
        last = view.tail(1).iloc[0]
        st.write({
            "Latest quarter": last["quarter"],
            "Expenses": float(last["expenses"]),
            "Eligible % (scenario)": float(last["eligible_pct_scn"]),
            "Eligible Expenses (scenario)": float(last["eligible_expenses_scn"]),
            "FFP Match (scenario)": float(last["ffp_match_value_scn"]),
        })

        st.success("Use this to show leadership how better coding/documentation can unlock additional federal revenue.")

with tabs[4]:
    st.subheader("Arrears Trends & Forecast (with scenario)")

    # Arrears trajectory indicator (last 12 months)
    start_ar = m_c["arrears_total"].iloc[-12]
    end_ar = m_c["arrears_total"].iloc[-1]
    ati = (end_ar - start_ar) / max(start_ar, 1e-9)

    s1, s2, s3 = st.columns(3)
    s1.metric("Arrears (start, 12 mo ago)", f"${start_ar:,.0f}")
    s2.metric("Arrears (current)", f"${end_ar:,.0f}")
    s3.metric("Arrears Trajectory (ATI)", f"{ati*100:.1f}%", "Improving" if ati < -0.05 else "Stable" if ati <= 0.05 else "Worsening")

    chart = m_c_s[["month","arrears_total","arrears_scn"]].set_index("month")
    st.line_chart(chart, height=320)

    st.markdown("**Arrears Growth Rate Forecast (demo approach):**")
    st.markdown(
        "- In production, forecasts come from R/Python models (ARIMA/Prophet/XGBoost) stored in a forecast table.\n"
        "- For this demo, the scenario adjusts collections and shows the implied arrears pressure over time."
    )

    st.info("Use this slide/tab to discuss early warning risk and proactive interventions (employment supports, outreach, payment plan optimization).")

with tabs[5]:
    st.subheader("Customer Transparency: Fee Breakdown + Payment Plan Calculator")

    left, right = st.columns([1,1])

    with left:
        st.markdown("### Fee Transparency Summary (typical payment)")
        avg = cust_c[["payment_amount","pass_through","arrears_applied","state_retained","fees"]].mean()
        breakdown = pd.DataFrame({
            "Component": ["Pass-through to family", "Applied to arrears", "State retained", "Fees"],
            "Amount": [avg["pass_through"], avg["arrears_applied"], avg["state_retained"], avg["fees"]]
        }).set_index("Component")
        st.bar_chart(breakdown, height=280)
        st.write("Plain-language example (demo):")
        st.write(
            f"For a typical payment of **${avg['payment_amount']:.0f}**, about **${avg['pass_through']:.0f}** goes to the family, "
            f"**${avg['arrears_applied']:.0f}** reduces arrears, **${avg['state_retained']:.0f}** is state retained (if applicable), "
            f"and **${avg['fees']:.0f}** covers fees."
        )

    with right:
        st.markdown("### Payment Plan Calculator (arrears payoff)")
        arrears = st.number_input("Starting arrears balance ($)", min_value=0.0, value=6000.0, step=100.0)
        monthly_payment = st.number_input("Monthly payment ($)", min_value=0.0, value=200.0, step=25.0)
        months_paid = st.slider("Payment stability (months paid out of last 12)", 0, 12, 8)

        if monthly_payment <= 0:
            st.warning("Enter a monthly payment greater than $0.")
        else:
            months_to_payoff = int(np.ceil(arrears / monthly_payment))
            # stability penalty (demo): if inconsistent, payoff takes longer
            stability_factor = 12 / max(months_paid, 1)
            adj_months = int(np.ceil(months_to_payoff * (0.7 + 0.3*stability_factor)))
            st.metric("Months to payoff (base)", f"{months_to_payoff}")
            st.metric("Months to payoff (stability-adjusted, demo)", f"{adj_months}")
            st.caption("In production, stability-adjustment can be driven by observed payment patterns and risk models.")

            # Scenario bars
            scen = pd.DataFrame({
                "Monthly Payment": [150, 200, 250, 300, 400],
                "Months to Payoff": [int(np.ceil(arrears/x)) if x>0 else np.nan for x in [150,200,250,300,400]]
            }).set_index("Monthly Payment")
            st.bar_chart(scen, height=260)

with tabs[6]:
    st.subheader("Data & Definitions (for transparency)")

    st.markdown("### Definitions")
    st.markdown(
        "**County Complexity Index (CCI):** Weighted composite of component scores (state avg = 1.0).\n"
        "**Expense Mix Ratio:** Share of spending by cost center (staff/enforcement/tech/outreach/admin).\n"
        "**FFP Match Value:** Eligible expenses × 66% (typical core IV-D).\n"
        "**Fiscal Pressure Score (FPS):** (Cost per case × CCI) ÷ Budget per case.\n"
        "**Arrears Trajectory Indicator (ATI):** % change in arrears over a defined window."
    )

    st.markdown("### Download demo data")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.download_button("county_complexity_scores.csv", complexity.to_csv(index=False), file_name="county_complexity_scores.csv")
    with col2:
        st.download_button("budget.csv", budget.to_csv(index=False), file_name="budget.csv")
    with col3:
        st.download_button("monthly_finance.csv", monthly.to_csv(index=False), file_name="monthly_finance.csv")
    with col4:
        st.download_button("quarterly_ffp.csv", quarterly.to_csv(index=False), file_name="quarterly_ffp.csv")
    with col5:
        st.download_button("customer_payment_breakdown.csv", customer.to_csv(index=False), file_name="customer_payment_breakdown.csv")

    st.markdown("### Notes for leadership demos")
    st.markdown(
        "- This is **mock data** designed for storytelling and usability testing.\n"
        "- In production, connect to your warehouse views and scheduled refresh (Alteryx/Tableau/SQL).\n"
        "- Keep the narrative focused on: **equity (CCI)**, **sustainability (FFP)**, **risk (arrears)**, and **strain (FPS)**."
    )