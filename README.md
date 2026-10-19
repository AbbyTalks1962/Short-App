# Equity Screening Lab

An interactive Streamlit coursework app for a three-stage financial-ratio screen, portfolio construction, and a configurable buy/sell shortlist.

## Important: no demo companies

This project does **not** include or silently load synthetic company rows. To run the analysis, either:

1. Load actual public-company annual statements through Financial Modeling Prep (FMP), or
2. Upload a verified real-company CSV using the required column names below.

The FMP connector requires an API key entered in the app. The key is not embedded in the source code or written to the repository. Use a plan that permits the required history and request volume. The screen requests eight annual gross-margin observations (seven year-over-year growth intervals) and financial statements for up to 1,000 companies; a free plan may not provide sufficient history or request capacity. Check the provider's current plan terms and data-use/licensing rules before publishing or redistributing financial data.

FMP official plan information: https://site.financialmodelingprep.com/pricing-plans

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Create an app at https://share.streamlit.io/ and select the repository, branch, and `app.py`.
3. Open the deployed app, enter an FMP API key, and click **Load real company universe**; or upload a verified CSV.
4. The app does not switch to fake data if a request fails. It shows an error instead.

## Data source and calculations

The live connector requests company information and annual income statements, balance sheets, and cash-flow statements from FMP's API. Companies without sufficient/accessible statements may be skipped. Consequently, a request for 1,000 companies can return fewer than 1,000 usable records; the app reports this rather than fabricating replacements.

- **STA (connector definition):** `(net income − operating cash flow) / average total assets`. Confirm this exact definition with your course material.
- **SNOA (connector definition):** `(operating assets − operating liabilities) / total assets`, where operating assets are total assets less cash and operating liabilities are total liabilities less interest-bearing short- and long-term debt. Missing inputs remain missing.
- **Altman Z:** standard public-company Z formula: `1.2 × working capital/assets + 1.4 × retained earnings/assets + 3.3 × EBIT/assets + 0.6 × market equity/total liabilities + sales/assets`. The standard formula has limitations across sectors, particularly financial firms; verify the version required in class.
- **GPA:** gross profit / total assets.
- **BTM:** book equity / market capitalization.
- **CFOA:** operating cash flow / total assets.
- **Margin observations:** gross profit margin (gross profit / revenue) for eight annual fiscal periods.
- **MS:** arithmetic average of seven annual margin-growth rates divided by their sample standard deviation.
- **MG:** seven-year annualized geometric growth in gross margin: `(ending margin / starting margin)^(1/7) − 1`. It requires positive start/end margins.
- **MM:** `max(percentile(MS), percentile(MG))`, with higher treated as better.

These ratio definitions are explicit implementation assumptions and should be reconciled with the instructor's exact definitions before the results are used for a graded submission. Missing data is not imputed with synthetic values.

## Screening workflow

- Stage 1: STA, SNOA, and Altman Z.
- Stage 2: MS, MG, and MM; previous stage ranks are included in cumulative ranking.
- Stage 3: GPA, BTM, and CFOA; previous stage ranks remain included.
- Each stage shows individual metric ranks, composite rank, eliminated companies, and downloadable CSV tables.
- Industry demeaning is optional independently for each stage. Ratio weights are configurable.
- Elimination percentages are configurable.
- Portfolio targets start equal across industries represented among final survivors. Industry targets can be edited and are normalized to 100%; each industry's allocation is split equally among its surviving companies.
- Buy/Sell: bottom-quintile flags for any of Z-score, GPA, or BTM prevent a company from entering the buy shortlist. The buy target percentage is adjustable; remaining companies are labelled SELL / EXCLUDE. The buy shortlist prioritizes high MM, high CFOA, low STA, and low SNOA.

## CSV upload schema

Required columns:

`Company, Ticker, Sector, Industry, STA, SNOA, Z_Score, GPA, BTM, CFOA, Margin_Tminus7, Margin_Tminus6, Margin_Tminus5, Margin_Tminus4, Margin_Tminus3, Margin_Tminus2, Margin_Tminus1, Margin_T`

For CSV uploads, supply actual, consistently defined metrics and eight annual gross-margin observations. The app computes MS, MG, and MM from the margin columns.

## Notes

This app is an educational screening tool, not investment advice. Historical financial statements may be revised, company fiscal years differ, and accounting ratios may not be comparable across all industries. Check formulas, data dates, missingness, and provider licensing.
