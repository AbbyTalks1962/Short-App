import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import time
from pathlib import Path

st.set_page_config(page_title="Equity Screening Lab", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1600px;}
[data-testid="stMetric"] {background: #f5f7fa; border: 1px solid #e5e9ef; padding: 12px 14px; border-radius: 10px;}
h1,h2,h3 {letter-spacing:-0.02em;}
</style>
""", unsafe_allow_html=True)

st.title("Equity Screening Lab")
st.caption("A transparent, adjustable three-stage equity screening and portfolio research tool")
MARGIN_COLS = [f"Margin_Tminus{i}" for i in range(7, 0, -1)] + ["Margin_T"]
BASE_REQUIRED = ["Company", "Ticker", "Sector", "Industry", "STA", "SNOA", "Z_Score", "GPA", "BTM", "CFOA"]

FMP_BASE = "https://financialmodelingprep.com/api/v3"

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fmp_get(url, params):
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and (payload.get("Error Message") or payload.get("error")):
        raise ValueError(payload.get("Error Message") or payload.get("error"))
    return payload

def _num(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None:
            try:
                value = float(value)
                if np.isfinite(value): return value
            except (TypeError, ValueError):
                pass
    return np.nan

def _safe_div(a, b):
    return a / b if pd.notna(a) and pd.notna(b) and b != 0 else np.nan

def _statement_rows(api_key, symbol, endpoint, limit=8):
    data = fmp_get(f"{FMP_BASE}/{endpoint}/{symbol}", {"period":"annual", "limit":limit, "apikey":api_key})
    if not isinstance(data, list): return []
    return sorted(data, key=lambda r: str(r.get("date", "")))

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_real_fmp_universe(api_key, target_count=1000):
    """Fetch real listed-company profiles and annual statements from Financial Modeling Prep.
    No synthetic fallback is used. Eight annual margin observations are required for seven growth intervals.
    """
    # Pull a wider screener pool first so we can diversify by sector/industry rather than take only mega-caps.
    pool = fmp_get(f"{FMP_BASE}/stock-screener", {
        "marketCapMoreThan": 100000000,
        "limit": min(5000, target_count * 4),
        "exchange": "NYSE,NASDAQ,AMEX",
        "apikey": api_key,
    })
    if not isinstance(pool, list) or not pool:
        raise ValueError("The provider returned no companies. Check the API key and plan access.")
    companies = pd.DataFrame(pool)
    symbol_col = "symbol" if "symbol" in companies else None
    name_col = "companyName" if "companyName" in companies else ("name" if "name" in companies else None)
    if not symbol_col or not name_col:
        raise ValueError("The provider screener response did not include expected company name and ticker fields.")
    companies = companies.rename(columns={symbol_col:"Ticker", name_col:"Company"})
    if "sector" not in companies: companies["sector"] = "Unknown"
    if "industry" not in companies: companies["industry"] = "Unknown"
    companies["Sector"] = companies["sector"].fillna("Unknown")
    companies["Industry"] = companies["industry"].fillna("Unknown")
    companies = companies.drop_duplicates("Ticker")
    # Take a diversified sample in screener order, cycling through industries.
    companies["_ord"] = np.arange(len(companies))
    companies = companies.sort_values(["Industry", "_ord"])
    companies["_industry_n"] = companies.groupby("Industry", sort=False).cumcount()
    companies = companies.sort_values(["_industry_n", "_ord"]).head(target_count)
    rows, failures = [], []
    progress = st.progress(0, text="Loading verified company financial statements…")
    status = st.empty()
    for idx, co in enumerate(companies.to_dict("records"), start=1):
        ticker = str(co["Ticker"])
        try:
            inc = _statement_rows(api_key, ticker, "income-statement", 8)
            bal = _statement_rows(api_key, ticker, "balance-sheet-statement", 8)
            cf = _statement_rows(api_key, ticker, "cash-flow-statement", 8)
            # Match by fiscal date; require eight margin observations and latest-year statements.
            inc_by = {str(r.get("date")): r for r in inc}
            bal_by = {str(r.get("date")): r for r in bal}
            cf_by = {str(r.get("date")): r for r in cf}
            dates = sorted(set(inc_by) & set(bal_by) & set(cf_by))
            if not dates:
                failures.append(ticker + ": no matching annual statements")
                continue
            latest_date = dates[-1]
            i, b, c = inc_by[latest_date], bal_by[latest_date], cf_by[latest_date]
            assets = _num(b, "totalAssets")
            prev_assets = _num(bal_by[dates[-2]], "totalAssets") if len(dates) > 1 else assets
            avg_assets = np.nanmean([assets, prev_assets]) if pd.notna(assets) and pd.notna(prev_assets) else assets
            revenue = _num(i, "revenue", "totalRevenue")
            gross_profit = _num(i, "grossProfit")
            net_income = _num(i, "netIncome")
            cfo = _num(c, "operatingCashFlow", "netCashProvidedByOperatingActivities")
            total_liabilities = _num(b, "totalLiabilities")
            current_assets = _num(b, "totalCurrentAssets")
            current_liabilities = _num(b, "totalCurrentLiabilities")
            retained = _num(b, "retainedEarnings")
            ebit = _num(i, "operatingIncome", "ebit")
            equity = _num(b, "totalStockholdersEquity", "totalStockholdersEquityIncludingMinorityInterest")
            cash = _num(b, "cashAndCashEquivalents", "cashAndShortTermInvestments")
            short_debt = _num(b, "shortTermDebt", "shortTermDebtAndCurrentPortionOfLongTermDebt")
            long_debt = _num(b, "longTermDebt")
            market_cap = _num(co, "marketCap")
            if not pd.notna(market_cap) or market_cap <= 0:
                # Screener normally supplies market cap; if not, use price x shares when available.
                market_cap = _num(co, "marketCap")
            # Standard public-company Altman Z formula; users should heed industry limitations in Methodology.
            wc = current_assets - current_liabilities if pd.notna(current_assets) and pd.notna(current_liabilities) else np.nan
            z = np.nan
            if all(pd.notna(v) for v in [assets, total_liabilities, wc, retained, ebit, market_cap, revenue]) and assets > 0 and total_liabilities > 0:
                z = 1.2*(wc/assets) + 1.4*(retained/assets) + 3.3*(ebit/assets) + 0.6*(market_cap/total_liabilities) + revenue/assets
            # STA: total accruals / average assets (lower is preferred).
            sta = _safe_div(net_income - cfo, avg_assets) if pd.notna(net_income) and pd.notna(cfo) else np.nan
            # SNOA: net operating assets / total assets; operating liabilities exclude interest-bearing debt.
            operating_assets = assets - cash if pd.notna(assets) and pd.notna(cash) else np.nan
            debt = ((short_debt if pd.notna(short_debt) else 0) + (long_debt if pd.notna(long_debt) else 0)) if (pd.notna(short_debt) or pd.notna(long_debt)) else np.nan
            operating_liabilities = total_liabilities - debt if pd.notna(total_liabilities) and pd.notna(debt) else np.nan
            snoa = _safe_div(operating_assets - operating_liabilities, assets) if pd.notna(operating_assets) and pd.notna(operating_liabilities) else np.nan
            gross_margin = _safe_div(gross_profit, revenue)
            row = {"Company":co["Company"], "Ticker":ticker, "Sector":co["Sector"], "Industry":co["Industry"],
                   "STA":sta, "SNOA":snoa, "Z_Score":z, "GPA":_safe_div(gross_profit, assets),
                   "BTM":_safe_div(equity, market_cap), "CFOA":_safe_div(cfo, assets), "MarketCap":market_cap,
                   "FiscalYear":latest_date[:4], "DataSource":"Financial Modeling Prep annual statements", "DataDate":latest_date}
            # Require eight historical gross-margin observations to calculate seven annual growth rates.
            margin_by_date = {}
            for date in dates:
                ir = inc_by[date]
                rev = _num(ir, "revenue", "totalRevenue")
                gp = _num(ir, "grossProfit")
                margin_by_date[date] = _safe_div(gp, rev)
            if len(margin_by_date) >= 8:
                last8 = [margin_by_date[dt] for dt in dates[-8:]]
                for j, value in enumerate(last8): row[f"Margin_Tminus{7-j}"] = value
                row["Margin_T"] = last8[-1]
                # The loop above creates Tminus0 for last point; normalize to Margin_T.
                row.pop("Margin_Tminus0", None)
            else:
                for col in [f"Margin_Tminus{k}" for k in range(7,0,-1)] + ["Margin_T"]: row[col] = np.nan
            rows.append(row)
        except Exception as exc:
            failures.append(f"{ticker}: {str(exc)[:120]}")
        if idx % 10 == 0 or idx == len(companies):
            progress.progress(idx / len(companies), text=f"Processed {idx:,} of {len(companies):,} companies…")
            status.caption(f"Valid companies loaded: {len(rows):,} · Unavailable/incomplete: {len(failures):,}")
    progress.empty(); status.empty()
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No complete company records could be loaded. Check API permissions and data availability.")
    result.attrs["failures"] = failures
    return result

def calculate_ms_mg(d):
    d = d.copy()
    cols = MARGIN_COLS
    if all(c in d.columns for c in cols):
        vals = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        # Annual margin growth rates from eight annual margin observations.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            growth = vals[:, 1:] / vals[:, :-1] - 1
        growth[~np.isfinite(growth)] = np.nan
        count = np.sum(np.isfinite(growth), axis=1)
        avg = np.divide(np.nansum(growth, axis=1), count, out=np.full(len(d), np.nan), where=count > 0)
        sd = np.nanstd(growth, axis=1, ddof=1)
        d["MS"] = np.divide(avg, sd, out=np.full(len(d), np.nan), where=np.isfinite(sd) & (sd > 0) & (count >= 2))
        start, end = vals[:, 0], vals[:, -1]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d["MG"] = np.where((start > 0) & (end > 0), (end / start) ** (1 / 7) - 1, np.nan)
    else:
        for c in ["MS", "MG"]:
            if c not in d: d[c] = np.nan
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def rank_series(s, high_good=False):
    # rank 1 = best; missing values are placed after valid observations
    return s.rank(method="average", ascending=high_good is False, na_option="bottom")

def safe_demean(frame, cols, enabled):
    frame = frame.copy()
    for col in cols:
        if col not in frame: continue
        adj = f"{col}_Adjusted"
        if enabled:
            frame[adj] = frame[col] - frame.groupby("Industry")[col].transform("mean")
        else:
            frame[adj] = frame[col]
    return frame

def weighted_rank_score(frame, rank_cols, weights):
    score = pd.Series(0.0, index=frame.index)
    total_weight = 0.0
    for col in rank_cols:
        w = float(weights.get(col.replace("R_", ""), 1.0))
        if w > 0:
            score += frame[col].fillna(frame[col].max() if frame[col].notna().any() else 0) * w
            total_weight += w
    return score / total_weight if total_weight else score

def run_screen(data, cuts, demean_flags, weights):
    d = calculate_ms_mg(data.copy())
    missing = [c for c in BASE_REQUIRED if c not in d.columns]
    if missing: raise ValueError("Missing required columns: " + ", ".join(missing))
    d = d.dropna(subset=["Company", "Ticker", "Sector", "Industry"]).drop_duplicates("Ticker").reset_index(drop=True)
    numeric = ["STA", "SNOA", "Z_Score", "MS", "MG", "GPA", "BTM", "CFOA"]
    for c in numeric: d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in MARGIN_COLS:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Disposition"] = "Survived all stages"
    stage_snapshots = {}
    removed_by_stage = {}

    def stage_rank(frame, metric_defs, demean, stage_num):
        x = safe_demean(frame, [m for m, _ in metric_defs], demean)
        rank_cols, metric_weights = [], {}
        for metric, high_good in metric_defs:
            source = f"{metric}_Adjusted" if demean else metric
            rcol = f"R_{metric}"
            x[rcol] = rank_series(x[source], high_good=high_good)
            rank_cols.append(rcol)
            metric_weights[metric] = weights.get(metric, 1.0)
        # Display an independent score/rank for the current stage.
        x[f"Stage{stage_num}_WeightedScore"] = weighted_rank_score(x, rank_cols, metric_weights)
        x[f"Stage{stage_num}_Rank"] = x[f"Stage{stage_num}_WeightedScore"].rank(method="min", ascending=True, na_option="bottom")
        return x, rank_cols

    def eliminate(frame, pct, score_col, stage_label):
        n = int(np.ceil(len(frame) * pct / 100)) if pct > 0 else 0
        order = frame.sort_values([score_col, "Ticker"], ascending=[False, True], na_position="first")
        removed = order.head(n)["Ticker"].tolist()
        kept = frame[~frame["Ticker"].isin(removed)].copy()
        return kept, removed

    # Stage 1: financial safety
    x1, r1 = stage_rank(d, [("STA", False), ("SNOA", False), ("Z_Score", True)], demean_flags[1], 1)
    x1["Stage1_CombinedScore"] = x1["Stage1_WeightedScore"]
    s1, removed1 = eliminate(x1, cuts[1], "Stage1_CombinedScore", "Stage 1")
    stage_snapshots["Stage 1"] = x1.copy(); removed_by_stage["Stage 1"] = removed1

    # Refresh all earlier ranks on the current survivor population before Stage 2.
    s1, r1 = stage_rank(s1, [("STA", False), ("SNOA", False), ("Z_Score", True)], demean_flags[1], 1)
    # Stage 2: show MS and MG ranks separately; MM is max of their within-population percentiles.
    x2 = safe_demean(s1, ["MS", "MG"], demean_flags[2])
    x2["R_MS"] = rank_series(x2["MS_Adjusted"] if demean_flags[2] else x2["MS"], high_good=True)
    x2["R_MG"] = rank_series(x2["MG_Adjusted"] if demean_flags[2] else x2["MG"], high_good=True)
    x2["MS_Percentile"] = (x2["R_MS"].max() - x2["R_MS"] + 1) / max(1, len(x2))
    x2["MG_Percentile"] = (x2["R_MG"].max() - x2["R_MG"] + 1) / max(1, len(x2))
    x2["MM"] = x2[["MS_Percentile", "MG_Percentile"]].max(axis=1, skipna=True)
    x2["R_MM"] = rank_series(x2["MM"], high_good=True)
    # weighted stage score uses the component ranks; MM is retained as the specified summary metric.
    wms, wmg = weights.get("MS", 1.0), weights.get("MG", 1.0)
    denom = wms + wmg
    x2["Stage2_MarginScore"] = ((x2["R_MS"] * wms + x2["R_MG"] * wmg) / denom) if denom else x2["R_MM"]
    x2["Stage2_CombinedScore"] = x2["Stage1_Rank"] + x2["Stage2_MarginScore"]
    x2["Stage2_Rank"] = x2["Stage2_CombinedScore"].rank(method="min", ascending=True, na_option="bottom")
    s2, removed2 = eliminate(x2, cuts[2], "Stage2_CombinedScore", "Stage 2")
    stage_snapshots["Stage 2"] = x2.copy(); removed_by_stage["Stage 2"] = removed2

    # Recalculate stages 1 and 2 ranks for the current population before Stage 3.
    s2, _ = stage_rank(s2, [("STA", False), ("SNOA", False), ("Z_Score", True)], demean_flags[1], 1)
    s2 = safe_demean(s2, ["MS", "MG"], demean_flags[2])
    s2["R_MS"] = rank_series(s2["MS_Adjusted"] if demean_flags[2] else s2["MS"], high_good=True)
    s2["R_MG"] = rank_series(s2["MG_Adjusted"] if demean_flags[2] else s2["MG"], high_good=True)
    s2["MS_Percentile"] = (s2["R_MS"].max() - s2["R_MS"] + 1) / max(1, len(s2))
    s2["MG_Percentile"] = (s2["R_MG"].max() - s2["R_MG"] + 1) / max(1, len(s2))
    s2["MM"] = s2[["MS_Percentile", "MG_Percentile"]].max(axis=1, skipna=True)
    s2["R_MM"] = rank_series(s2["MM"], high_good=True)
    denom = weights.get("MS", 1.0) + weights.get("MG", 1.0)
    s2["Stage2_MarginScore"] = ((s2["R_MS"] * weights.get("MS", 1.0) + s2["R_MG"] * weights.get("MG", 1.0)) / denom) if denom else s2["R_MM"]
    s2["Stage2_CombinedScore"] = s2["Stage1_Rank"] + s2["Stage2_MarginScore"]
    s2["Stage2_Rank"] = s2["Stage2_CombinedScore"].rank(method="min", ascending=True, na_option="bottom")

    # Stage 3: profitability/valuation metrics
    x3, r3 = stage_rank(s2, [("GPA", True), ("BTM", True), ("CFOA", True)], demean_flags[3], 3)
    x3["Stage3_CombinedScore"] = x3["Stage2_CombinedScore"] + x3["Stage3_WeightedScore"]
    x3["Stage3_Rank"] = x3["Stage3_CombinedScore"].rank(method="min", ascending=True, na_option="bottom")
    s3, removed3 = eliminate(x3, cuts[3], "Stage3_CombinedScore", "Stage 3")
    stage_snapshots["Stage 3"] = x3.copy(); removed_by_stage["Stage 3"] = removed3

    # Build complete audit/disposition.
    audit = d.copy()
    for stage, ids in removed_by_stage.items():
        audit.loc[audit["Ticker"].isin(ids), "Disposition"] = f"Eliminated at {stage}"
    # Final survivor ranks are refreshed on the final population for presentation.
    if len(s3):
        s3, _ = stage_rank(s3, [("STA", False), ("SNOA", False), ("Z_Score", True)], demean_flags[1], 1)
        s3 = safe_demean(s3, ["MS", "MG"], demean_flags[2])
        s3["R_MS"] = rank_series(s3["MS_Adjusted"] if demean_flags[2] else s3["MS"], high_good=True)
        s3["R_MG"] = rank_series(s3["MG_Adjusted"] if demean_flags[2] else s3["MG"], high_good=True)
        s3["MS_Percentile"] = (s3["R_MS"].max() - s3["R_MS"] + 1) / max(1, len(s3))
        s3["MG_Percentile"] = (s3["R_MG"].max() - s3["R_MG"] + 1) / max(1, len(s3))
        s3["MM"] = s3[["MS_Percentile", "MG_Percentile"]].max(axis=1, skipna=True)
        s3["R_MM"] = rank_series(s3["MM"], high_good=True)
        denom = weights.get("MS", 1.0) + weights.get("MG", 1.0)
        s3["Stage2_MarginScore"] = ((s3["R_MS"] * weights.get("MS", 1.0) + s3["R_MG"] * weights.get("MG", 1.0)) / denom) if denom else s3["R_MM"]
        s3["Stage2_CombinedScore"] = s3["Stage1_Rank"] + s3["Stage2_MarginScore"]
        s3["Stage2_Rank"] = s3["Stage2_CombinedScore"].rank(method="min", ascending=True, na_option="bottom")
        s3, _ = stage_rank(s3, [("GPA", True), ("BTM", True), ("CFOA", True)], demean_flags[3], 3)
        s3["Stage3_CombinedScore"] = s3["Stage2_CombinedScore"] + s3["Stage3_WeightedScore"]
        s3["Final_Rank"] = s3["Stage3_CombinedScore"].rank(method="min", ascending=True, na_option="bottom")
        s3 = s3.sort_values(["Final_Rank", "Ticker"])
    return d, stage_snapshots, s1, s2, s3, audit, removed_by_stage

with st.sidebar:
    st.header("Screen settings")
    st.caption("Elimination percentages apply sequentially to the surviving population.")
    cuts = {1: st.slider("Stage 1 elimination (%)", 0, 50, 10, 5), 2: st.slider("Stage 2 elimination (%)", 0, 50, 10, 5), 3: st.slider("Stage 3 elimination (%)", 0, 50, 10, 5)}
    st.divider()
    st.subheader("Industry demeaning")
    demean_flags = {
        1: st.checkbox("Demean Stage 1 ratios by industry", True),
        2: st.checkbox("Demean Stage 2 ratios by industry", False),
        3: st.checkbox("Demean Stage 3 ratios by industry", False),
    }
    st.caption("Demeaning subtracts the current population's industry mean from each selected ratio before ranking.")
    st.divider()
    st.subheader("Ratio weights")
    st.caption("Set each ratio's relative importance. A weight of 0 excludes it from the weighted score.")
    weights = {}
    for metric, default in [("STA",1.0),("SNOA",1.0),("Z_Score",1.0),("MS",1.0),("MG",1.0),("GPA",1.0),("BTM",1.0),("CFOA",1.0)]:
        weights[metric] = st.slider(f"{metric} weight", 0.0, 5.0, default, 0.25, key=f"weight_{metric}")
    st.divider()
    st.subheader("Real company data")
    st.caption("No demo-company dataset is used. Load real annual statements from Financial Modeling Prep or upload a verified real-data CSV.")
    api_key = st.text_input("Financial Modeling Prep API key", type="password", help="Keep this key private. It is used only to request company data.")
    target_count = st.number_input("Target number of real companies", min_value=100, max_value=1000, value=1000, step=100)
    st.session_state["target_count"] = int(target_count)
    fetch_real = st.button("Load real company universe", type="primary", use_container_width=True)
    uploaded = st.file_uploader("Or upload a verified real-company CSV", type=["csv"], help="Required columns: Company, Ticker, Sector, Industry, STA, SNOA, Z_Score, GPA, BTM, CFOA, plus eight annual gross-margin observations.")
    st.caption("Seven-year MS/MG needs eight annual observations. FMP's free tier may not provide enough history or request capacity; check your plan before loading 1,000 companies.")

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        data_source = "User-uploaded real-company data — verify provider and fiscal periods"
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}"); st.stop()
elif fetch_real:
    if not api_key.strip():
        st.error("Enter your Financial Modeling Prep API key first."); st.stop()
    try:
        with st.spinner("Connecting to Financial Modeling Prep and retrieving real company financial statements…"):
            data = load_real_fmp_universe(api_key.strip(), int(target_count))
        data_source = "Financial Modeling Prep API — actual company annual statements"
        st.session_state["real_company_data"] = data
        st.session_state["real_company_source"] = data_source
        st.session_state["real_company_failures"] = data.attrs.get("failures", [])
    except Exception as e:
        st.error(f"Could not load real financial data: {e}")
        st.info("Check your API key, plan permissions, request limits, and internet access. No synthetic data will be substituted.")
        st.stop()
elif "real_company_data" in st.session_state:
    data = st.session_state["real_company_data"].copy()
    data_source = st.session_state.get("real_company_source", "Financial Modeling Prep API")
else:
    st.info("No companies are loaded yet. Enter an FMP API key and click **Load real company universe**, or upload a verified real-company CSV. This app will not populate demo companies.")
    st.stop()

try:
    original, snapshots, s1, s2, final_stage, audit, removed = run_screen(data, cuts, demean_flags, weights)
except Exception as e:
    st.error(f"Could not run the screen: {e}")
    st.info("Required columns: Company, Ticker, Sector, Industry, STA, SNOA, Z_Score, GPA, BTM, CFOA, plus eight annual margin columns (Margin_Tminus7 through Margin_T) or pre-calculated MS and MG.")
    st.stop()

n0 = len(original); n1 = n0 - len(removed["Stage 1"]); n2 = n1 - len(removed["Stage 2"]); n3 = len(final_stage)
a,b,c,dcol = st.columns(4)
a.metric("Starting companies", f"{n0:,}")
b.metric("After Stage 1", f"{n1:,}", delta=f"-{len(removed['Stage 1'])} removed")
c.metric("After Stage 2", f"{n2:,}", delta=f"-{len(removed['Stage 2'])} removed")
dcol.metric("Final survivors", f"{n3:,}", delta=f"-{len(removed['Stage 3'])} removed")
st.caption(f"Data: {data_source} · Rank 1 is best · All ratios are displayed individually in the stage tables.")
if "Financial Modeling Prep" in data_source and n0 < int(st.session_state.get("target_count", 1000)):
    st.warning(f"Only {n0:,} companies with usable real financial statements were returned. Missing or inaccessible statements are not replaced with demo data.")
if "real_company_failures" in st.session_state and st.session_state["real_company_failures"]:
    with st.expander(f"Data retrieval notes ({len(st.session_state['real_company_failures']):,})"):
        st.write("Some companies could not be loaded because statements were missing, incomplete, or inaccessible on the provider plan.")
        st.code("\n".join(st.session_state["real_company_failures"][:100]))

# Portfolio allocation targets by industry, equal weights by default.
st.header("Portfolio construction")
st.caption("Industry targets start equally weighted. Edit the target weights to tilt toward or away from particular industries; weights are normalized to 100%.")
if n3:
    industries = sorted(final_stage["Industry"].dropna().unique().tolist())
    default_weight = 100.0 / max(1, len(industries))
    target_df = pd.DataFrame({"Industry": industries, "Target_Weight_%": [default_weight] * len(industries)})
    edited_targets = st.data_editor(target_df, use_container_width=True, hide_index=True, key="industry_targets", column_config={"Industry": st.column_config.TextColumn(disabled=True), "Target_Weight_%": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0, format="%.2f%%")})
    total_target = edited_targets["Target_Weight_%"].sum()
    if total_target <= 0:
        st.warning("Set at least one industry target above zero to calculate portfolio weights.")
        final_stage["Portfolio_Weight"] = 0.0
    else:
        target_map = (edited_targets.set_index("Industry")["Target_Weight_%"] / total_target).to_dict()
        counts = final_stage.groupby("Industry")["Ticker"].transform("count")
        final_stage["Industry_Target_Weight"] = final_stage["Industry"].map(target_map).fillna(0.0)
        final_stage["Portfolio_Weight"] = final_stage["Industry_Target_Weight"] / counts
    alloc = final_stage.groupby("Industry", as_index=False).agg(Companies=("Ticker", "count"), Target_Weight=("Industry_Target_Weight", "first"), Actual_Weight=("Portfolio_Weight", "sum"))
    alloc["Target_Weight"] = (alloc["Target_Weight"] * 100).round(2)
    alloc["Actual_Weight"] = (alloc["Actual_Weight"] * 100).round(2)
    left,right = st.columns([1,1])
    with left: st.dataframe(alloc.sort_values("Actual_Weight", ascending=False), use_container_width=True, hide_index=True)
    with right:
        fig = px.bar(alloc, x="Industry", y="Actual_Weight", title="Final portfolio weight by industry", hover_data=["Companies", "Target_Weight"])
        fig.update_layout(xaxis_title="", yaxis_title="Portfolio weight (%)")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No companies survived. Reduce one or more elimination percentages.")

tabs = st.tabs(["Overview", "Stage 1 — Safety", "Stage 2 — Margins", "Stage 3 — Profitability", "Final portfolio", "Buy / Sell", "Company database", "Methodology & data"])
with tabs[0]:
    l,r = st.columns(2)
    with l:
        st.subheader("Survivors by sector")
        if n3:
            sc = final_stage["Sector"].value_counts().rename_axis("Sector").reset_index(name="Companies")
            st.plotly_chart(px.bar(sc, x="Sector", y="Companies", color="Sector"), use_container_width=True)
    with r:
        funnel = pd.DataFrame({"Stage":["Starting universe","After Stage 1","After Stage 2","Final survivors"],"Companies":[n0,n1,n2,n3]})
        st.plotly_chart(px.funnel(funnel, x="Companies", y="Stage"), use_container_width=True)
    st.subheader("Top final survivors")
    if n3:
        showcols = [c for c in ["Final_Rank","Company","Ticker","Sector","Industry","R_STA","R_SNOA","R_Z_Score","R_MS","R_MG","MM","R_GPA","R_BTM","R_CFOA","Stage3_CombinedScore","Portfolio_Weight"] if c in final_stage]
        st.dataframe(final_stage[showcols].head(25), use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Stage 1 — Financial safety")
    st.markdown("STA and SNOA: lower is better. Z-score: higher is better. Toggle industry demeaning and ratio weights in the sidebar.")
    t = snapshots["Stage 1"]
    cols = [c for c in ["Company","Ticker","Sector","Industry","STA","STA_Adjusted","R_STA","SNOA","SNOA_Adjusted","R_SNOA","Z_Score","Z_Score_Adjusted","R_Z_Score","Stage1_WeightedScore","Stage1_Rank"] if c in t]
    st.dataframe(t[cols].sort_values("Stage1_Rank"), use_container_width=True, hide_index=True)
    st.download_button("Download Stage 1 table", t[cols].to_csv(index=False).encode(), "stage1_safety.csv", "text/csv")
    st.subheader("Stage 1 eliminated")
    st.dataframe(t[t["Ticker"].isin(removed["Stage 1"])][cols], use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Stage 2 — Margin of safety and growth")
    st.markdown("MS = average annual growth in margin divided by the sample standard deviation of annual margin growth. MG = seven-year geometric annualized margin growth. MM = max(percentile(MS), percentile(MG)). Higher is treated as better.")
    t = snapshots["Stage 2"]
    cols = [c for c in ["Company","Ticker","Sector","Industry","MS","MS_Adjusted","R_MS","MS_Percentile","MG","MG_Adjusted","R_MG","MG_Percentile","MM","R_MM","Stage1_Rank","Stage2_MarginScore","Stage2_CombinedScore","Stage2_Rank"] if c in t]
    st.dataframe(t[cols].sort_values("Stage2_Rank"), use_container_width=True, hide_index=True)
    st.download_button("Download Stage 2 table", t[cols].to_csv(index=False).encode(), "stage2_margins.csv", "text/csv")
    st.subheader("Stage 2 eliminated")
    st.dataframe(t[t["Ticker"].isin(removed["Stage 2"])][cols], use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Stage 3 — Profitability and valuation")
    st.markdown("GPA, BTM and CFOA are shown separately. Current default: higher values rank better for all three; confirm the BTM direction against your course methodology.")
    t = snapshots["Stage 3"]
    cols = [c for c in ["Company","Ticker","Sector","Industry","GPA","GPA_Adjusted","R_GPA","BTM","BTM_Adjusted","R_BTM","CFOA","CFOA_Adjusted","R_CFOA","Stage2_CombinedScore","Stage3_WeightedScore","Stage3_CombinedScore","Stage3_Rank"] if c in t]
    st.dataframe(t[cols].sort_values("Stage3_Rank"), use_container_width=True, hide_index=True)
    st.download_button("Download Stage 3 table", t[cols].to_csv(index=False).encode(), "stage3_profitability.csv", "text/csv")
    st.subheader("Stage 3 eliminated")
    st.dataframe(t[t["Ticker"].isin(removed["Stage 3"])][cols], use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Final portfolio")
    if n3:
        st.metric("Total portfolio weight", f"{final_stage['Portfolio_Weight'].sum()*100:.2f}%")
        cols = [c for c in ["Final_Rank","Company","Ticker","Sector","Industry","STA","SNOA","Z_Score","MS","MG","MM","GPA","BTM","CFOA","R_STA","R_SNOA","R_Z_Score","R_MS","R_MG","R_GPA","R_BTM","R_CFOA","Portfolio_Weight"] if c in final_stage]
        st.dataframe(final_stage[cols], use_container_width=True, hide_index=True)
        st.download_button("Download final portfolio CSV", final_stage[cols].to_csv(index=False).encode(), "final_portfolio.csv", "text/csv")
    else: st.info("No survivors under the current settings.")
    st.caption("Portfolio weights are mathematical allocations for coursework, not investment advice.")

with tabs[5]:
    st.subheader("Buy / Sell decisions")
    st.markdown("This is a configurable coursework screen, not investment advice. The sell screen flags low Altman Z-score, GPA, and BTM; the buy screen favors high MM and CFOA and low STA and SNOA.")
    buy_pct = st.slider("Target percentage to BUY", 1, 99, 20, 1, key="buy_pct")
    sell_pct = 100 - buy_pct
    st.metric("Target percentage to SELL / exclude", f"{sell_pct}%")
    st.caption("Companies in the bottom quintile or with missing data for any sell metric are excluded from the buy shortlist. The buy percentage is a maximum target; the actual buy bucket can be smaller if many companies fail the sell screen.")
    if n3:
        trade = final_stage.copy()
        # Sell score: higher rank number means worse on all three sell criteria (low raw values are undesirable).
        trade["Sell_Z_Rank"] = rank_series(trade["Z_Score"], high_good=True)
        trade["Sell_GPA_Rank"] = rank_series(trade["GPA"], high_good=True)
        trade["Sell_BTM_Rank"] = rank_series(trade["BTM"], high_good=True)
        trade["Sell_Score"] = trade[["Sell_Z_Rank", "Sell_GPA_Rank", "Sell_BTM_Rank"]].mean(axis=1)
        trade["Buy_MM_Rank"] = rank_series(trade["MM"], high_good=True)
        trade["Buy_CFOA_Rank"] = rank_series(trade["CFOA"], high_good=True)
        trade["Buy_STA_Rank"] = rank_series(trade["STA"], high_good=False)
        trade["Buy_SNOA_Rank"] = rank_series(trade["SNOA"], high_good=False)
        trade["Buy_Score"] = trade[["Buy_MM_Rank", "Buy_CFOA_Rank", "Buy_STA_Rank", "Buy_SNOA_Rank"]].mean(axis=1)
        # Low values on any sell metric trigger a sell/exclusion flag before the buy shortlist is selected.
        for metric in ["Z_Score", "GPA", "BTM"]:
            trade[f"Low_{metric}_Flag"] = trade[metric].isna() | (trade[metric] <= trade[metric].quantile(0.20) if trade[metric].notna().any() else True)
        trade["Sell_Flag"] = trade[["Low_Z_Score_Flag", "Low_GPA_Flag", "Low_BTM_Flag"]].any(axis=1)
        trade["Buy_Rank"] = trade["Buy_Score"].rank(method="first", ascending=True)
        trade["Sell_Rank"] = trade["Sell_Score"].rank(method="first", ascending=False)
        n_buy_target = max(1, int(np.ceil(len(trade) * buy_pct / 100)))
        buy_eligible = trade[~trade["Sell_Flag"]].copy()
        # The target is up to buy_pct of the final universe; companies flagged by the sell metrics cannot be bought.
        buy_names = set(buy_eligible.nsmallest(n_buy_target, "Buy_Score")["Ticker"])
        trade["Trade_Action"] = np.where(trade["Ticker"].isin(buy_names), "BUY", "SELL / EXCLUDE")
        st.write(f"Buy bucket: {len(buy_names):,} companies · Sell/exclude bucket: {len(trade)-len(buy_names):,} companies · Sell-metric flagged: {int(trade['Sell_Flag'].sum()):,}")
        show = ["Trade_Action", "Buy_Rank", "Sell_Rank", "Company", "Ticker", "Sector", "Industry", "MM", "STA", "SNOA", "CFOA", "Z_Score", "GPA", "BTM", "Buy_Score", "Sell_Score", "Sell_Flag"]
        show = [c for c in show if c in trade]
        action_filter = st.selectbox("Show actions", ["All", "BUY", "SELL / EXCLUDE"])
        shown = trade if action_filter == "All" else trade[trade["Trade_Action"] == action_filter]
        st.dataframe(shown[show].sort_values("Buy_Rank"), use_container_width=True, hide_index=True)
        st.download_button("Download buy/sell decisions", trade[show].to_csv(index=False).encode(), "buy_sell_decisions.csv", "text/csv")
        st.subheader("Companies flagged by sell metrics")
        st.caption("Flags identify companies in the bottom quintile of any sell metric. They are displayed separately from the adjustable buy/sell split.")
        st.dataframe(trade[trade["Sell_Flag"]][["Company", "Ticker", "Z_Score", "GPA", "BTM", "Low_Z_Score_Flag", "Low_GPA_Flag", "Low_BTM_Flag"]], use_container_width=True, hide_index=True)
    else:
        st.info("No final portfolio companies are available to evaluate.")

with tabs[6]:
    st.subheader("Company database")
    f1,f2 = st.columns(2)
    sectors = sorted(original["Sector"].dropna().unique().tolist())
    selected = f1.selectbox("Filter sector", ["All sectors"] + sectors)
    search = f2.text_input("Search company or ticker")
    view = original.copy()
    if selected != "All sectors": view = view[view["Sector"] == selected]
    if search: view = view[view["Company"].astype(str).str.contains(search, case=False, na=False) | view["Ticker"].astype(str).str.contains(search, case=False, na=False)]
    st.write(f"{len(view):,} companies shown")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download company database CSV", original.to_csv(index=False).encode(), "company_universe.csv", "text/csv")

with tabs[7]:
    st.subheader("Methodology and data requirements")
    st.markdown("""
**Data source:** no synthetic company universe is bundled or loaded automatically. Load actual annual statements through the Financial Modeling Prep API or upload a verified real-company CSV. The API key is not stored in the project files. Review provider access, fiscal dates, and data licensing before use.

**Industry demeaning:** optional independently for each stage. When enabled, each ratio is adjusted by subtracting its industry mean among the population being ranked at that stage.

**Seven-year window:** eight annual margin observations (T−7 through T) create seven year-over-year growth rates.

**MS:** arithmetic average of annual margin growth rates divided by their sample standard deviation.

**MG:** annualized geometric growth in margin: `(ending margin / starting margin)^(1/7) − 1`. This formula requires positive beginning and ending margins; if margins are zero or negative, MG is left missing rather than producing a misleading value.

**MM:** `max(percentile(MS), percentile(MG))`. MS and MG are also displayed and weighted separately so the analyst can control their contribution.

**Portfolio allocation:** initial target weights are equal across the industries represented among final survivors. Edit the target percentage by industry; the app normalizes all nonzero targets to 100%, then allocates each industry's target equally among that industry's surviving companies.

**Required data columns:** `Company, Ticker, Sector, Industry, STA, SNOA, Z_Score, GPA, BTM, CFOA`, plus eight annual gross-margin observations: `Margin_Tminus7` through `Margin_Tminus1` and `Margin_T`. The FMP connector derives the ratios from annual statements; for CSV uploads, supply the derived metrics and margin observations.

**Ratio direction defaults:** STA and SNOA lower is better; Z-score higher is better; MS/MG/MM higher is better; GPA, BTM, and CFOA higher is better for the screening stage. The trade decision stage separately treats lower Z-score, GPA, and BTM as sell signals. Confirm course definitions. The standard Altman Z formula is not equally suitable for every sector, particularly financial firms.
""")
    template = pd.DataFrame(columns=BASE_REQUIRED + MARGIN_COLS)
    st.download_button("Download real-data CSV template", template.to_csv(index=False).encode(), "company_data_template.csv", "text/csv")
    st.warning("No demo-company data is included. Only actual API responses or uploaded real-company data are screened.")

st.divider()
st.caption("Educational analysis tool only. Check formula definitions, fiscal-year alignment, missing data and data-provider licensing before relying on results.")
