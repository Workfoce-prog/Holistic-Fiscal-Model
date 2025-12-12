# Child Support Fiscal Leadership Demo (Streamlit)

This is an **unbranded**, government-professional Streamlit demo app showing a **Holistic Fiscal Metrics Framework**
for child support programs.

## What it demonstrates
- County Complexity Index (CCI) with component drivers
- Expense Mix Ratio (where dollars go)
- Federal Match (FFP) with a coding-quality scenario slider
- Arrears trends and scenario-based pressure
- Fiscal Pressure Score (FPS) as an equity-aware strain indicator
- Customer-facing fee transparency and a payment plan payoff calculator

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo controls
Use the sidebar to:
- Select a county
- Apply an unemployment shock (demo affects collections and arrears)
- Adjust IV-D coding quality (demo affects eligible % and match value)

## Data
All data in `/data` is mock and safe for demonstrations.