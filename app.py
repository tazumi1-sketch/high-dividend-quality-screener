from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
DEMO_DATA = BASE / "data" / "demo_data.csv"
UNIVERSE = BASE / "data" / "universe.csv"

st.set_page_config(
    page_title="高配当優良株スクリーナー",
    page_icon="💴",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 3rem; max-width: 1050px;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.small-note {font-size:.86rem;color:#6b7280;}
@media (max-width: 640px) {
  .block-container {padding-left:.65rem;padding-right:.65rem;padding-top:.65rem;}
  h1 {font-size:1.65rem !important;} h2 {font-size:1.25rem !important;}
  [data-testid="stMetricValue"] {font-size:1.25rem;}
}
</style>
""",
    unsafe_allow_html=True,
)

NUMERIC_COLUMNS = [
    "price", "dividend_yield", "market_cap_bil", "equity_ratio", "debt_equity",
    "operating_margin", "net_income_bil", "free_cash_flow_bil", "payout_ratio",
    "dividend_cuts_5y", "dividend_maintained_years", "eps", "per",
]


def as_num(value):
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def pct(value):
    """Normalize Yahoo ratios (0.05) and already-percent values (5.0)."""
    value = as_num(value)
    if pd.isna(value):
        return np.nan
    return value * 100 if abs(value) <= 1.5 else value


def score_one(row: pd.Series, cfg: dict) -> dict:
    y = as_num(row.get("dividend_yield"))
    cap = as_num(row.get("market_cap_bil"))
    eq = as_num(row.get("equity_ratio"))
    de = as_num(row.get("debt_equity"))
    margin = as_num(row.get("operating_margin"))
    profit = as_num(row.get("net_income_bil"))
    fcf = as_num(row.get("free_cash_flow_bil"))
    payout = as_num(row.get("payout_ratio"))
    cuts = as_num(row.get("dividend_cuts_5y"))
    years = as_num(row.get("dividend_maintained_years"))
    eps = as_num(row.get("eps"))
    per = as_num(row.get("per"))

    # 配当魅力度 30点
    if pd.notna(y) and cfg["yield_min"] <= y <= cfg["yield_max"]:
        yield_pts = 15
    elif pd.notna(y) and 4.0 <= y <= 7.0:
        yield_pts = 12
    elif pd.notna(y) and 3.5 <= y <= 8.0:
        yield_pts = 7
    else:
        yield_pts = 0
    payout_pts = 10 if pd.notna(payout) and 30 <= payout <= 60 else 7 if pd.notna(payout) and 20 <= payout <= 75 else 2 if pd.notna(payout) and 0 <= payout <= 90 else 0
    history_pts = 5 if cuts == 0 and years >= 5 else 3 if pd.notna(cuts) and cuts <= 1 and years >= 3 else 0
    dividend_score = yield_pts + payout_pts + history_pts

    # 財務健全性 35点
    cap_pts = 8 if pd.notna(cap) and cap >= cfg["market_cap_min"] else 4 if pd.notna(cap) and cap >= 300 else 0
    eq_pts = 8 if pd.notna(eq) and eq >= 50 else 6 if pd.notna(eq) and eq >= cfg["equity_min"] else 2 if pd.notna(eq) and eq >= 20 else 0
    de_pts = 7 if pd.notna(de) and de <= 50 else 5 if pd.notna(de) and de <= 100 else 2 if pd.notna(de) and de <= 150 else 0
    margin_pts = 5 if pd.notna(margin) and margin >= 10 else 3 if pd.notna(margin) and margin >= 5 else 0
    profit_pts = 4 if pd.notna(profit) and profit > 0 else 0
    fcf_pts = 3 if pd.notna(fcf) and fcf > 0 else 0
    finance_score = cap_pts + eq_pts + de_pts + margin_pts + profit_pts + fcf_pts

    # 配当継続力 35点
    cut_pts = 12 if cuts == 0 else 6 if cuts == 1 else 0
    years_pts = 8 if pd.notna(years) and years >= 5 else 5 if pd.notna(years) and years >= 3 else 0
    eps_pts = 5 if pd.notna(eps) and eps > 0 else 0
    safe_payout_pts = 5 if pd.notna(payout) and 20 <= payout <= 70 else 2 if pd.notna(payout) and 0 <= payout <= 90 else 0
    per_pts = 5 if pd.notna(per) and 0 < per <= 15 else 3 if pd.notna(per) and 0 < per <= 20 else 0
    continuity_score = cut_pts + years_pts + eps_pts + safe_payout_pts + per_pts

    total = dividend_score + finance_score + continuity_score
    red_flags = []
    if pd.notna(payout) and payout > 90:
        red_flags.append("配当性向90%超")
    if pd.notna(profit) and profit <= 0:
        red_flags.append("最終赤字")
    if pd.notna(fcf) and fcf <= 0:
        red_flags.append("FCF赤字")
    if pd.notna(cuts) and cuts >= 2:
        red_flags.append("5年で複数回減配")
    if pd.notna(y) and y > 8:
        red_flags.append("異常高利回り")
    if pd.notna(eq) and eq < 20:
        red_flags.append("自己資本比率20%未満")

    gate_checks = [
        pd.notna(y) and cfg["yield_min"] <= y <= cfg["yield_max"],
        pd.notna(cap) and cap >= cfg["market_cap_min"],
        pd.notna(profit) and profit > 0,
        pd.isna(payout) or payout <= 90,
        pd.isna(cuts) or cuts <= 1,
        pd.isna(eq) or eq >= cfg["equity_min"],
    ]
    gates = int(sum(gate_checks))
    if total >= 80 and gates == len(gate_checks) and not red_flags:
        rank = "A かなり安定"
    elif total >= 65 and gates >= 5 and len(red_flags) <= 1:
        rank = "B 安定候補"
    else:
        rank = "C 要確認"

    missing = sum(pd.isna(as_num(row.get(c))) for c in NUMERIC_COLUMNS[1:])
    return {
        "配当点": dividend_score,
        "財務点": finance_score,
        "継続力点": continuity_score,
        "総合点": total,
        "必須通過": f"{gates}/{len(gate_checks)}",
        "ランク": rank,
        "警告": "・".join(red_flags) if red_flags else "なし",
        "欠損数": missing,
    }


def score_frame(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = raw.copy()
    for col in NUMERIC_COLUMNS:
        if col not in d.columns:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    for col in ["code", "ticker", "company"]:
        if col not in d.columns:
            d[col] = ""
        d[col] = d[col].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    scored = d.apply(lambda r: pd.Series(score_one(r, cfg)), axis=1)
    d = pd.concat([d, scored], axis=1)
    order = pd.Categorical(d["ランク"], ["A かなり安定", "B 安定候補", "C 要確認"], ordered=True)
    d["_rank_order"] = order
    return d.sort_values(["_rank_order", "総合点", "dividend_yield"], ascending=[True, False, False]).drop(columns="_rank_order")


def completed_dividend_stats(dividends: pd.Series) -> tuple[float, float]:
    if dividends is None or len(dividends) == 0:
        return np.nan, np.nan
    s = dividends.copy()
    try:
        years = s.groupby(s.index.year).sum().sort_index()
    except Exception:
        return np.nan, np.nan
    current_year = datetime.now(timezone.utc).year
    years = years[years.index < current_year].tail(6)
    if len(years) < 2:
        return np.nan, float(len(years))
    cuts = int((years.pct_change().dropna() < -0.05).sum())
    maintained = 1
    for i in range(len(years) - 1, 0, -1):
        if years.iloc[i] >= years.iloc[i - 1] * 0.95:
            maintained += 1
        else:
            break
    return float(cuts), float(maintained)


def fetch_one(ticker: str, code: str, fallback_name: str) -> dict:
    import yfinance as yf

    obj = yf.Ticker(ticker)
    info = obj.info or {}
    cuts, maintained = completed_dividend_stats(obj.dividends)
    return {
        "code": code,
        "ticker": ticker,
        # 銘柄一覧に登録した日本語名を画面表示とCSV出力で統一して使う。
        "company": fallback_name or info.get("longName") or info.get("shortName") or code,
        "price": as_num(info.get("currentPrice") or info.get("regularMarketPrice")),
        "dividend_yield": pct(info.get("dividendYield")),
        "market_cap_bil": as_num(info.get("marketCap")) / 1e9,
        "equity_ratio": np.nan,
        "debt_equity": as_num(info.get("debtToEquity")),
        "operating_margin": pct(info.get("operatingMargins")),
        "net_income_bil": as_num(info.get("netIncomeToCommon")) / 1e9,
        "free_cash_flow_bil": as_num(info.get("freeCashflow")) / 1e9,
        "payout_ratio": pct(info.get("payoutRatio")),
        "dividend_cuts_5y": cuts,
        "dividend_maintained_years": maintained,
        "eps": as_num(info.get("trailingEps")),
        "per": as_num(info.get("trailingPE") or info.get("forwardPE")),
        "data_as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": "Yahoo Finance (yfinance)",
    }


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_online(selected_codes: tuple[str, ...]) -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE, dtype=str)
    universe = universe[universe["code"].isin(selected_codes)]
    rows = []
    bar = st.progress(0, text="データ取得を開始します")
    for n, (_, item) in enumerate(universe.iterrows(), start=1):
        bar.progress((n - 1) / max(len(universe), 1), text=f"{item['code']} {item['company']} を取得中")
        try:
            rows.append(fetch_one(item["ticker"], item["code"], item["company"]))
        except Exception as exc:
            rows.append({"code": item["code"], "ticker": item["ticker"], "company": item["company"], "source": f"取得失敗: {type(exc).__name__}"})
        time.sleep(0.15)
    bar.progress(1.0, text="取得が完了しました")
    return pd.DataFrame(rows)


@st.cache_data
def load_demo() -> pd.DataFrame:
    return pd.read_csv(DEMO_DATA, dtype={"code": str})


@st.cache_data
def load_universe() -> pd.DataFrame:
    return pd.read_csv(UNIVERSE, dtype=str)


st.title("高配当・優良株スクリーナー")
st.caption("配当の高さ × 財務健全性 × 配当継続力を100点で判定")
st.warning("売買推奨ではありません。配当・業績予想は変わるため、購入前に会社の最新IR資料で必ず再確認してください。")

with st.sidebar:
    st.header("判定条件")
    yield_min = st.slider("配当利回り 下限（%）", 3.0, 7.0, 4.5, 0.1)
    yield_max = st.slider("配当利回り 上限（%）", 4.0, 10.0, 6.5, 0.1)
    market_cap_min = st.select_slider("最低時価総額（億円）", [1000, 3000, 5000, 10000, 30000], value=5000)
    equity_min = st.slider("自己資本比率 下限（%）", 10, 70, 30, 5)
    st.caption("金融・商社などは業種特性により、自己資本比率や負債指標の単純比較が適さない場合があります。")

cfg = {
    "yield_min": yield_min,
    "yield_max": max(yield_max, yield_min),
    "market_cap_min": market_cap_min / 10,  # 億円 → 10億円単位
    "equity_min": equity_min,
}

source_tab, screen_tab, detail_tab, rules_tab = st.tabs(["📥 データ", "📊 候補", "🔎 個別", "📋 判定条件"])

with source_tab:
    st.subheader("データを選ぶ")
    mode = st.radio("使用するデータ", ["説明用サンプル", "CSVを読み込む", "オンライン取得"], horizontal=False)
    raw = None
    if mode == "説明用サンプル":
        raw = load_demo()
        st.info("画面確認用の架空データです。実在銘柄の投資判断には使えません。")
    elif mode == "CSVを読み込む":
        uploaded = st.file_uploader("所定形式のCSV", type=["csv"])
        st.download_button("CSVひな形をダウンロード", data=(BASE / "data" / "input_template.csv").read_bytes(), file_name="high_dividend_input_template.csv", mime="text/csv")
        if uploaded:
            raw = pd.read_csv(uploaded, dtype={"code": str})
    else:
        universe = load_universe()
        choices = [f"{r.code} {r.company}" for r in universe.itertuples()]
        default = choices[:20]
        picked = st.multiselect("取得する銘柄（最初は20銘柄程度を推奨）", choices, default=default)
        st.caption("無料データを使用するため、欠損・遅延・取得制限があります。自己資本比率は欠損になることがあります。")
        if st.button("選択銘柄の最新データを取得", type="primary", disabled=not picked):
            codes = tuple(x.split(" ", 1)[0] for x in picked)
            try:
                raw = fetch_online(codes)
                st.session_state["online_data"] = raw
            except ImportError:
                st.error("yfinance がありません。requirements.txt を使って再起動してください。")
        elif "online_data" in st.session_state:
            raw = st.session_state["online_data"]

    if raw is not None and len(raw):
        st.session_state["screening_raw"] = raw
        scored_preview = score_frame(raw, cfg)
        c1, c2, c3 = st.columns(3)
        c1.metric("銘柄数", len(scored_preview))
        c2.metric("Aランク", int(scored_preview["ランク"].str.startswith("A").sum()))
        c3.metric("Bランク", int(scored_preview["ランク"].str.startswith("B").sum()))
        st.success("データを読み込みました。「候補」タブで確認できます。")

raw_active = st.session_state.get("screening_raw", load_demo())
scored = score_frame(raw_active, cfg)

with screen_tab:
    st.subheader("候補銘柄")
    rank_filter = st.multiselect("表示ランク", ["A かなり安定", "B 安定候補", "C 要確認"], default=["A かなり安定", "B 安定候補"])
    no_warning = st.checkbox("警告なしだけ表示", value=False)
    view = scored[scored["ランク"].isin(rank_filter)].copy()
    if no_warning:
        view = view[view["警告"].eq("なし")]
    st.metric("該当銘柄数", len(view))
    show_cols = ["code", "company", "dividend_yield", "market_cap_bil", "payout_ratio", "dividend_cuts_5y", "総合点", "ランク", "警告"]
    st.dataframe(
        view[show_cols], hide_index=True, width="stretch",
        column_config={
            "code": "コード", "company": "会社名",
            "dividend_yield": st.column_config.NumberColumn("配当利回り", format="%.2f%%"),
            "market_cap_bil": st.column_config.NumberColumn("時価総額(10億円)", format="%.0f"),
            "payout_ratio": st.column_config.NumberColumn("配当性向", format="%.1f%%"),
            "dividend_cuts_5y": st.column_config.NumberColumn("5年減配回数", format="%.0f"),
            "総合点": st.column_config.ProgressColumn("総合点", min_value=0, max_value=100, format="%d"),
        },
    )
    st.download_button("判定結果CSVをダウンロード", data=scored.drop(columns=[], errors="ignore").to_csv(index=False).encode("utf-8-sig"), file_name="high_dividend_screening.csv", mime="text/csv")

with detail_tab:
    st.subheader("個別銘柄の判定")
    q = st.text_input("証券コードまたは会社名", placeholder="例：9432 / NTT")
    if q.strip():
        mask = scored["code"].str.contains(q, case=False, regex=False) | scored["company"].str.contains(q, case=False, regex=False)
        hits = scored[mask]
        if hits.empty:
            st.warning("読み込み済みデータに該当銘柄がありません。")
        else:
            labels = [f"{r.code} {r.company}" for r in hits.itertuples()]
            chosen = st.selectbox("検索結果", labels)
            r = hits.iloc[labels.index(chosen)]
            st.markdown(f"### {r['code']}　{r['company']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("総合", f"{int(r['総合点'])}点")
            c2.metric("配当", f"{int(r['配当点'])}/30")
            c3.metric("財務", f"{int(r['財務点'])}/35")
            c4.metric("継続力", f"{int(r['継続力点'])}/35")
            st.info(f"判定：{r['ランク']}　｜　警告：{r['警告']}　｜　必須条件：{r['必須通過']}")
            details = pd.DataFrame([
                ["予想配当利回り", r["dividend_yield"], f"{cfg['yield_min']:.1f}〜{cfg['yield_max']:.1f}%"],
                ["時価総額(10億円)", r["market_cap_bil"], f"{cfg['market_cap_min']:.0f}以上"],
                ["自己資本比率", r["equity_ratio"], f"{cfg['equity_min']}%以上"],
                ["負債/自己資本", r["debt_equity"], "100%以下を優先"],
                ["営業利益率", r["operating_margin"], "5%以上を優先"],
                ["配当性向", r["payout_ratio"], "20〜70%を優先"],
                ["5年減配回数", r["dividend_cuts_5y"], "0回を優先"],
                ["配当維持年数", r["dividend_maintained_years"], "5年以上を優先"],
                ["PER", r["per"], "20倍以下を優先"],
            ], columns=["項目", "実データ", "目安"])
            st.dataframe(details, hide_index=True, width="stretch")
    else:
        st.caption("先に「データ」タブで対象データを読み込んでください。")

with rules_tab:
    st.subheader("100点の内訳")
    st.markdown("""
| 分野 | 配点 | 主な評価項目 |
|---|---:|---|
| 配当魅力度 | 30 | 利回り、配当性向、減配歴 |
| 財務健全性 | 35 | 時価総額、自己資本、負債、利益率、利益、FCF |
| 配当継続力 | 35 | 減配回数、維持年数、EPS、配当余力、PER |
""")
    st.markdown("**Aランク**は80点以上・必須6条件通過・警告なし。**Bランク**は65点以上を基本とし、危険な高配当株を警告表示します。")
    st.caption("銀行・保険・証券・商社・REITは財務構造が異なるため、同業種内比較を併用してください。")

st.divider()
st.caption("無料データには誤り・欠損・遅延があります。最終確認は決算短信、有価証券報告書、配当予想修正などの一次資料で行ってください。")
