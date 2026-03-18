"""
股票監控儀表板 - 完整修復版 v2.0
==================================
已修復問題清單：
 1. matched_rank 未定義 NameError → 先初始化為 None
 2. while True + time.sleep() → time.sleep() + st.rerun()（Streamlit Cloud 相容）
 3. @st.cache_data 接收不可哈希 DataFrame → 改用 ticker/period/interval 字串作 key
 4. VIX merge 時區不符全為 NaN → 統一 tz_localize(None)
 5. make_subplots 中 yaxis2/yaxis3 overlaying 衝突 → 改用 4 行獨立子圖
 6. px.line()["data"][0] 不規範 → 全用 go.Scatter
 7. generate_comprehensive_interpretation dense_desc 在 return 後無效 → 前移
 8. VWAP 跨日累積錯誤 → 按日分組計算
 9. 新增完整回測系統（組合信號勝率分析）
10. send_email_alert 參數過多 → 整合為 dict
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from itertools import combinations
import time
import traceback

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 股票監控儀表板", layout="wide", page_icon="📈")
load_dotenv()

SENDER_EMAIL    = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "")

try:
    BOT_TOKEN      = st.secrets["telegram"]["BOT_TOKEN"]
    CHAT_ID        = st.secrets["telegram"]["CHAT_ID"]
    telegram_ready = True
except Exception:
    BOT_TOKEN = CHAT_ID = None
    telegram_ready = False

# ── Sell-signal set (used in success-rate & backtest direction logic) ─────────
SELL_SIGNALS = {
    "📉 High<Low","📉 MACD賣出","📉 EMA賣出","📉 價格趨勢賣出","📉 價格趨勢賣出(量)",
    "📉 價格趨勢賣出(量%)","📉 普通跳空(下)","📉 突破跳空(下)","📉 持續跳空(下)",
    "📉 衰竭跳空(下)","📉 連續向下賣出","📉 SMA50下降趨勢","📉 SMA50_200下降趨勢",
    "📉 新卖出信号","📉 RSI-MACD Overbought Crossover","📉 EMA-SMA Downtrend Sell",
    "📉 Volume-MACD Sell","📉 EMA10_30賣出","📉 EMA10_30_40強烈賣出","📉 看跌吞沒",
    "📉 烏雲蓋頂","📉 上吊線","📉 黃昏之星","📉 VWAP賣出","📉 MFI熊背離賣出",
    "📉 OBV突破賣出","📉 VIX恐慌賣出","📉 VIX上升趨勢賣出",
}

# ═════════════════════════════════════════════════════════════════════════════
#  INDICATOR FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def send_telegram_alert(msg: str) -> bool:
    if not (BOT_TOKEN and CHAT_ID):
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg,
                   "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.get(url, params=payload, timeout=10)
        return r.status_code == 200 and r.json().get("ok", False)
    except Exception:
        return False


def calculate_macd(df, fast=12, slow=26, signal=9):
    e1 = df["Close"].ewm(span=fast, adjust=False).mean()
    e2 = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = e1 - e2
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def calculate_rsi(df, periods=14):
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    FIX: group by calendar date so VWAP resets each day.
    Works for both intraday and daily data.
    """
    df2 = df.copy()
    df2["_dt"] = pd.to_datetime(df2["Datetime"]).dt.date
    typical = (df2["High"] + df2["Low"] + df2["Close"]) / 3
    tp_vol  = typical * df2["Volume"]

    vwap_vals = []
    for date, grp in df2.groupby("_dt", sort=False):
        cum_tv = tp_vol.loc[grp.index].cumsum()
        cum_v  = df2.loc[grp.index, "Volume"].cumsum().replace(0, np.nan)
        vwap_vals.append(cum_tv / cum_v)

    result = pd.concat(vwap_vals).reindex(df2.index)
    return result


def calculate_mfi(df, periods=14):
    typical    = (df["High"] + df["Low"] + df["Close"]) / 3
    mf         = typical * df["Volume"]
    pos_mf     = mf.where(typical > typical.shift(1), 0).rolling(window=periods).sum()
    neg_mf     = mf.where(typical < typical.shift(1), 0).rolling(window=periods).sum()
    mfi        = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))
    return mfi.fillna(50)


def calculate_obv(df):
    return (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()


def calculate_volume_profile(df, bins=50, window=100, top_n=3):
    n = min(len(df), window)
    recent = df.tail(n).copy()
    pmin, pmax = recent["Low"].min(), recent["High"].max()
    if pmax == pmin:
        return []
    edges   = np.linspace(pmin, pmax, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    profile = np.zeros(bins)
    for _, row in recent.iterrows():
        lo_i = max(0, int(np.searchsorted(edges, row["Low"],  "left")  - 1))
        hi_i = max(0, min(int(np.searchsorted(edges, row["High"], "right") - 1), bins - 1))
        lo_i = min(lo_i, bins - 1)
        span = hi_i - lo_i + 1
        for j in range(lo_i, hi_i + 1):
            profile[j] += row["Volume"] / span
    top_idx = np.argsort(profile)[-top_n:][::-1]
    return [{"price_center": centers[i], "volume": profile[i],
             "price_low": edges[i], "price_high": edges[i + 1]}
            for i in top_idx if profile[i] > 0]


def get_vix_data(period, interval):
    try:
        vdf = yf.Ticker("^VIX").history(period=period, interval=interval).reset_index()
        if vdf.empty:
            return pd.DataFrame()
        if "Date" in vdf.columns:
            vdf = vdf.rename(columns={"Date": "Datetime"})
        # FIX: strip timezone before merge
        vdf["Datetime"]      = pd.to_datetime(vdf["Datetime"]).dt.tz_localize(None)
        vdf["VIX_Change_Pct"]= vdf["Close"].pct_change().round(4) * 100
        return vdf[["Datetime", "Close", "VIX_Change_Pct"]].rename(columns={"Close": "VIX"})
    except Exception:
        return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
#  CACHED K-LINE PATTERN (FIX: use hashable params, not DataFrame)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_kline_patterns(ticker: str, period: str, interval: str,
                       body_ratio: float, shadow_ratio: float, doji_body: float,
                       _cache_buster: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval).reset_index()
    if df.empty:
        return pd.DataFrame(columns=["Datetime","K線形態","單根解讀"])
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "Datetime"})
    df["Datetime"]  = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
    df["前5均量"]   = df["Volume"].rolling(window=5).mean()

    patterns, interps = [], []
    for idx, row in df.iterrows():
        p, t = _classify_kline(row, idx, df, body_ratio, shadow_ratio, doji_body)
        patterns.append(p); interps.append(t)
    df["K線形態"]  = patterns
    df["單根解讀"] = interps
    return df[["Datetime","K線形態","單根解讀"]]


def _classify_kline(row, idx, df, body_ratio, shadow_ratio, doji_body):
    p, t = "普通K線", "波動有限，方向不明顯"
    if idx == 0:
        return p, t
    po, pc = df["Open"].iloc[idx-1], df["Close"].iloc[idx-1]
    ph, pl = df["High"].iloc[idx-1], df["Low"].iloc[idx-1]
    co, cc, ch, cl = row["Open"], row["Close"], row["High"], row["Low"]
    body   = abs(cc - co)
    rng    = ch - cl if ch != cl else 1e-9
    hi_vol = row["Volume"] > row.get("前5均量", 0)
    is_up  = df["Close"].iloc[max(0, idx-5):idx].mean() < cc if idx >= 5 else False
    is_dn  = df["Close"].iloc[max(0, idx-5):idx].mean() > cc if idx >= 5 else False
    lower  = min(co, cc) - cl
    upper  = ch - max(co, cc)

    if body < rng*0.3 and lower >= shadow_ratio*max(body,1e-9) and upper < lower and is_dn:
        p = "錘子線"; t = "下方支撐，多方承接" + ("，放量增強" if hi_vol else "")
    elif body < rng*0.3 and upper >= shadow_ratio*max(body,1e-9) and lower < upper and is_up:
        p = "射擊之星"; t = "高位拋壓" + ("，放量賣出" if hi_vol else "")
    elif body < doji_body*rng:
        p = "十字星"; t = "市場猶豫，方向未明"
    elif cc > co and body > body_ratio*rng:
        p = "大陽線"; t = "多方強勢" + ("，放量有力" if hi_vol else "")
    elif cc < co and body > body_ratio*rng:
        p = "大陰線"; t = "空方強勢" + ("，放量偏空" if hi_vol else "")
    elif cc > co and pc < po and co < pc and cc > po and hi_vol:
        p = "看漲吞噬"; t = "陽線包覆前日陰線，預示反轉"
    elif cc < co and pc > po and co > pc and cc < po and hi_vol:
        p = "看跌吞噬"; t = "陰線包覆前日陽線，預示反轉"
    elif is_up and cc < co and pc > po and co > pc and cc < (po+pc)/2:
        p = "烏雲蓋頂"; t = "上升趨勢中陰線壓制，賣壓加重"
    elif is_dn and cc > co and pc < po and co < pc and cc > (po+pc)/2:
        p = "刺透形態"; t = "下跌趨勢中陽線反攻，買方介入"
    elif (idx > 1 and
          df["Close"].iloc[idx-2] < df["Open"].iloc[idx-2] and
          abs(df["Close"].iloc[idx-1]-df["Open"].iloc[idx-1]) < 0.3*abs(df["Close"].iloc[idx-2]-df["Open"].iloc[idx-2]) and
          cc > co and cc > (po+pc)/2 and hi_vol):
        p = "早晨之星"; t = "下跌後強陽，預示反轉"
    elif (idx > 1 and
          df["Close"].iloc[idx-2] > df["Open"].iloc[idx-2] and
          abs(df["Close"].iloc[idx-1]-df["Open"].iloc[idx-1]) < 0.3*abs(df["Close"].iloc[idx-2]-df["Open"].iloc[idx-2]) and
          cc < co and cc < (po+pc)/2 and hi_vol):
        p = "黃昏之星"; t = "上漲後強陰，預示反轉"
    return p, t


# ═════════════════════════════════════════════════════════════════════════════
#  EMAIL (FIX: consolidated dict param)
# ═════════════════════════════════════════════════════════════════════════════

def send_email_alert(ticker: str, price_pct: float, volume_pct: float, active_signals: dict):
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        return
    desc = {
        "macd_buy":"📈 MACD買入","macd_sell":"📉 MACD賣出",
        "ema_buy":"📈 EMA買入","ema_sell":"📉 EMA賣出",
        "new_buy":"📈 新買入","new_sell":"📉 新賣出",
        "vwap_buy":"📈 VWAP買入","vwap_sell":"📉 VWAP賣出",
        "mfi_bull":"📈 MFI牛背離","mfi_bear":"📉 MFI熊背離",
        "obv_buy":"📈 OBV突破買入","obv_sell":"📉 OBV突破賣出",
        "vix_panic":"📉 VIX恐慌賣出","vix_calm":"📈 VIX平靜買入",
        "bullish_eng":"📈 看漲吞沒","bearish_eng":"📉 看跌吞沒",
        "morning_star":"📈 早晨之星","evening_star":"📉 黃昏之星",
        "hammer":"📈 錘頭線","hanging_man":"📉 上吊線",
    }
    lines = [f"股票: {ticker}", f"股價變動: {price_pct:.2f}%",
             f"成交量變動: {volume_pct:.2f}%", ""]
    for k, label in desc.items():
        if active_signals.get(k):
            lines.append(label)
    lines.append("\n⚠️ 系統偵測到異動，請立即查看市場情況。")
    msg = MIMEMultipart()
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT_EMAIL
    msg["Subject"] = f"📣 股票異動通知：{ticker}"
    msg.attach(MIMEText("\n".join(lines), "plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(SENDER_EMAIL, SENDER_PASSWORD)
            srv.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        st.toast(f"📬 Email 已發送")
    except Exception as e:
        st.error(f"Email 發送失敗：{e}")


# ═════════════════════════════════════════════════════════════════════════════
#  BACKTEST: combination win-rate
# ═════════════════════════════════════════════════════════════════════════════

def backtest_signal_combinations(df: pd.DataFrame, min_combo=2,
                                  max_combo=4, min_occ=3) -> pd.DataFrame:
    """
    For every combination of signals (size min_combo..max_combo),
    compute the win-rate = P(next_close > current_close | all signals present).
    """
    df = df.copy()
    df["_next_up"] = df["Close"].shift(-1) > df["Close"]
    signal_sets = []
    for marks in df["異動標記"].fillna(""):
        sigs = {s.strip() for s in str(marks).split(", ") if s.strip() and "🔥" not in s}
        signal_sets.append(sigs)
    df["_sigs"] = signal_sets

    all_s = set()
    for s in signal_sets:
        all_s.update(s)
    all_s = sorted(all_s)

    rows = []
    for r in range(min_combo, min(max_combo + 1, len(all_s) + 1)):
        for combo in combinations(all_s, r):
            cs   = set(combo)
            mask = df["_sigs"].apply(lambda s: cs.issubset(s))
            sub  = df[mask]
            if len(sub) < min_occ:
                continue
            sell_n = sum(1 for s in combo if s in SELL_SIGNALS)
            is_sell = sell_n > len(combo) / 2
            wr = (1 - sub["_next_up"].mean()) * 100 if is_sell else sub["_next_up"].mean() * 100
            rows.append({
                "信號組合":  " + ".join(combo),
                "信號數量":  r,
                "勝率(%)":   round(wr, 1),
                "出現次數":  len(sub),
                "方向":      "做空" if is_sell else "做多",
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("勝率(%)", ascending=False).head(30)


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL MARKING  (core logic, returns comma-joined string)
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_signals(data: pd.DataFrame,
                        params: dict) -> pd.Series:
    """
    Vectorised-friendly signal marker.
    Returns pd.Series of strings aligned to data index.
    """
    results = []
    for idx, row in data.iterrows():
        results.append(_mark_one(row, idx, data, params))
    return pd.Series(results, index=data.index)


def _prev(data, col, idx, n=1):
    i = idx - n
    if i < 0:
        return np.nan
    return data[col].iloc[i]


def _mark_one(row, idx, data, p):
    sigs = []
    macd = row["MACD"]
    rsi  = row["RSI"]
    fma  = row["前5均量"] if pd.notna(row["前5均量"]) else 0

    def pv(col, n=1):
        return _prev(data, col, idx, n)

    # ── 量價 ──────────────────────────────────────────────────────────────────
    pa = row.get("📈 股價漲跌幅(%)", np.nan)
    va = row.get("📊 成交量變動幅(%)", np.nan)
    if pd.notna(pa) and pd.notna(va) and abs(pa) >= p["PRICE_TH"] and abs(va) >= p["VOLUME_TH"]:
        sigs.append("✅ 量價")

    # ── Low>High / High<Low ───────────────────────────────────────────────────
    if idx > 0:
        if row["Low"]  > pv("High"):  sigs.append("📈 Low>High")
        if row["High"] < pv("Low"):   sigs.append("📉 High<Low")

    # ── Close position ────────────────────────────────────────────────────────
    cnh = row.get("Close_N_High", np.nan)
    cnl = row.get("Close_N_Low",  np.nan)
    if pd.notna(cnh) and cnh >= p["HIGH_N_HIGH_TH"]:  sigs.append("📈 HIGH_N_HIGH")
    if pd.notna(cnl) and cnl >= p["LOW_N_LOW_TH"]:    sigs.append("📉 LOW_N_LOW")

    # ── MACD ──────────────────────────────────────────────────────────────────
    if idx > 0:
        if macd > 0  and pv("MACD") <= 0 and rsi < 50: sigs.append("📈 MACD買入")
        if macd <= 0 and pv("MACD") > 0  and rsi > 50: sigs.append("📉 MACD賣出")

    # ── EMA5/10 ───────────────────────────────────────────────────────────────
    if idx > 0:
        if row["EMA5"] > row["EMA10"] and pv("EMA5") <= pv("EMA10") and rsi < 50:
            sigs.append("📈 EMA買入")
        if row["EMA5"] < row["EMA10"] and pv("EMA5") >= pv("EMA10") and rsi > 50:
            sigs.append("📉 EMA賣出")

    # ── Price trend ───────────────────────────────────────────────────────────
    if idx > 0:
        ph, pl, pc = pv("High"), pv("Low"), pv("Close")
        vc = row.get("Volume Change %", 0) or 0
        if row["High"] > ph and row["Low"] > pl and row["Close"] > pc:
            if macd > 0:                              sigs.append("📈 價格趨勢買入")
            if row["Volume"] > fma and rsi < 50:      sigs.append("📈 價格趨勢買入(量)")
            if vc > 15 and rsi < 50:                  sigs.append("📈 價格趨勢買入(量%)")
        if row["High"] < ph and row["Low"] < pl and row["Close"] < pc:
            if macd < 0:                              sigs.append("📉 價格趨勢賣出")
            if row["Volume"] > fma and rsi > 50:      sigs.append("📉 價格趨勢賣出(量)")
            if vc > 15 and rsi > 50:                  sigs.append("📉 價格趨勢賣出(量%)")

    # ── Gaps ──────────────────────────────────────────────────────────────────
    if idx > 0:
        gap_pct = (row["Open"] - pv("Close")) / (pv("Close") or 1) * 100
        is_up   = gap_pct >  p["GAP_TH"]
        is_dn   = gap_pct < -p["GAP_TH"]
        if is_up or is_dn:
            window5 = data["Close"].iloc[max(0, idx-5):idx]
            trend   = window5.mean() if len(window5) else row["Close"]
            prev5   = data["Close"].iloc[max(0, idx-6):idx-1].mean() if idx >= 6 else trend
            hi_vol  = row["Volume"] > fma
            reversal = (idx < len(data)-1 and
                        ((is_up and data["Close"].iloc[idx+1] < row["Close"]) or
                         (is_dn and data["Close"].iloc[idx+1] > row["Close"])))
            if is_up:
                if reversal and hi_vol:
                    sigs.append("📈 衰竭跳空(上)")
                elif row["Close"] > trend > prev5 and hi_vol:
                    sigs.append("📈 持續跳空(上)")
                elif row["High"] > data["High"].iloc[max(0,idx-5):idx].max() and hi_vol:
                    sigs.append("📈 突破跳空(上)")
                else:
                    sigs.append("📈 普通跳空(上)")
            else:
                if reversal and hi_vol:
                    sigs.append("📉 衰竭跳空(下)")
                elif row["Close"] < trend < prev5 and hi_vol:
                    sigs.append("📉 持續跳空(下)")
                elif row["Low"] < data["Low"].iloc[max(0,idx-5):idx].min() and hi_vol:
                    sigs.append("📉 突破跳空(下)")
                else:
                    sigs.append("📉 普通跳空(下)")

    # ── Continuous ────────────────────────────────────────────────────────────
    if row.get("Continuous_Up",   0) >= p["CONT_UP"]   and rsi < 70: sigs.append("📈 連續向上買入")
    if row.get("Continuous_Down", 0) >= p["CONT_DOWN"]  and rsi > 30: sigs.append("📉 連續向下賣出")

    # ── SMA50/200 ─────────────────────────────────────────────────────────────
    if pd.notna(row.get("SMA50")):
        if row["Close"] > row["SMA50"] and macd > 0:   sigs.append("📈 SMA50上升趨勢")
        elif row["Close"] < row["SMA50"] and macd < 0: sigs.append("📉 SMA50下降趨勢")
    if pd.notna(row.get("SMA50")) and pd.notna(row.get("SMA200")):
        if row["Close"] > row["SMA50"] > row["SMA200"] and macd > 0:   sigs.append("📈 SMA50_200上升趨勢")
        elif row["Close"] < row["SMA50"] < row["SMA200"] and macd < 0: sigs.append("📉 SMA50_200下降趨勢")

    # ── New buy/sell ──────────────────────────────────────────────────────────
    if idx > 0:
        pc = pv("Close")
        if row["Close"] > row["Open"] > pc and rsi < 70: sigs.append("📈 新买入信号")
        if row["Close"] < row["Open"] < pc and rsi > 30: sigs.append("📉 新卖出信号")

    # ── Pivot ─────────────────────────────────────────────────────────────────
    pr = row.get("Price Change %", 0) or 0
    vc_ = row.get("Volume Change %", 0) or 0
    if abs(pr) > p["PC_TH"] and abs(vc_) > p["VC_TH"] and macd > row.get("Signal_Line", 0):
        sigs.append("🔄 新转折点")
    if len(sigs) > 8:
        sigs.append(f"🔥 關鍵轉折({len(sigs)}信號)")

    # ── RSI-MACD composite ────────────────────────────────────────────────────
    if idx > 0:
        if rsi < 30 and macd > 0 and pv("MACD") <= 0:   sigs.append("📈 RSI-MACD Oversold Crossover")
        if rsi > 70 and macd < 0 and pv("MACD") >= 0:   sigs.append("📉 RSI-MACD Overbought Crossover")

    # ── EMA-SMA trend ─────────────────────────────────────────────────────────
    s50 = row.get("SMA50", np.nan)
    if pd.notna(s50):
        if row["EMA5"] > row["EMA10"] and row["Close"] > s50: sigs.append("📈 EMA-SMA Uptrend Buy")
        if row["EMA5"] < row["EMA10"] and row["Close"] < s50: sigs.append("📉 EMA-SMA Downtrend Sell")

    # ── Volume-MACD ───────────────────────────────────────────────────────────
    if idx > 0:
        if row["Volume"] > fma and macd > 0 and pv("MACD") <= 0: sigs.append("📈 Volume-MACD Buy")
        if row["Volume"] > fma and macd < 0 and pv("MACD") >= 0: sigs.append("📉 Volume-MACD Sell")

    # ── EMA 10/30/40 ─────────────────────────────────────────────────────────
    if idx > 0:
        if row["EMA10"] > row["EMA30"] and pv("EMA10") <= pv("EMA30"):
            sigs.append("📈 EMA10_30買入")
            if row["EMA10"] > row.get("EMA40", 0): sigs.append("📈 EMA10_30_40強烈買入")
        if row["EMA10"] < row["EMA30"] and pv("EMA10") >= pv("EMA30"):
            sigs.append("📉 EMA10_30賣出")
            if row["EMA10"] < row.get("EMA40", 999999): sigs.append("📉 EMA10_30_40強烈賣出")

    # ── Candlestick patterns ───────────────────────────────────────────────────
    if idx > 0:
        co, cc, ch, cl = row["Open"], row["Close"], row["High"], row["Low"]
        po, pc2 = pv("Open"), pv("Close")
        body = abs(cc - co); rng = ch - cl if ch != cl else 1e-9
        hi_vol = row["Volume"] > fma
        lower = min(co, cc) - cl; upper = ch - max(co, cc)

        if pc2 < po and cc > co and co < pc2 and cc > po and hi_vol and rsi < 50:
            sigs.append("📈 看漲吞沒")
        if pc2 > po and cc < co and co > pc2 and cc < po and hi_vol and rsi > 50:
            sigs.append("📉 看跌吞沒")
        if body < rng*0.3 and lower >= 2*max(body,1e-9) and upper < lower and hi_vol and rsi < 50:
            sigs.append("📈 錘頭線")
        if body < rng*0.3 and lower >= 2*max(body,1e-9) and upper < lower and hi_vol and rsi > 50:
            sigs.append("📉 上吊線")
        if pc2 > po and co > pc2 and cc < co and cc < (po+pc2)/2 and hi_vol:
            sigs.append("📉 烏雲蓋頂")
        if pc2 < po and co < pc2 and cc > co and cc > (po+pc2)/2 and hi_vol:
            sigs.append("📈 刺透形態")

    if idx > 1:
        p2o, p2c = data["Open"].iloc[idx-2], data["Close"].iloc[idx-2]
        p1o, p1c = data["Open"].iloc[idx-1], data["Close"].iloc[idx-1]
        co2, cc2 = row["Open"], row["Close"]
        hi_vol = row["Volume"] > fma
        if p2c < p2o and abs(p1c-p1o) < 0.3*abs(p2c-p2o) and cc2 > co2 and cc2 > (p2o+p2c)/2 and hi_vol and rsi < 50:
            sigs.append("📈 早晨之星")
        if p2c > p2o and abs(p1c-p1o) < 0.3*abs(p2c-p2o) and cc2 < co2 and cc2 < (p2o+p2c)/2 and hi_vol and rsi > 50:
            sigs.append("📉 黃昏之星")

    # ── Breakout ──────────────────────────────────────────────────────────────
    if idx > 0 and pd.notna(row.get("High_Max")) and row["High"] > data["High_Max"].iloc[idx-1]:
        sigs.append("📈 BreakOut_5K")
    if idx > 0 and pd.notna(row.get("Low_Min")) and row["Low"] < data["Low_Min"].iloc[idx-1]:
        sigs.append("📉 BreakDown_5K")

    # ── VWAP ─────────────────────────────────────────────────────────────────
    if idx > 0 and pd.notna(row.get("VWAP")) and pd.notna(pv("VWAP")):
        if row["Close"] > row["VWAP"] and pv("Close") <= pv("VWAP"):  sigs.append("📈 VWAP買入")
        elif row["Close"] < row["VWAP"] and pv("Close") >= pv("VWAP"): sigs.append("📉 VWAP賣出")

    # ── MFI divergence ────────────────────────────────────────────────────────
    w = p["MFI_WIN"]
    if idx >= w:
        if data.get("MFI_Bull_Div") is not None and data["MFI_Bull_Div"].iloc[idx]:
            sigs.append("📈 MFI牛背離買入")
        if data.get("MFI_Bear_Div") is not None and data["MFI_Bear_Div"].iloc[idx]:
            sigs.append("📉 MFI熊背離賣出")

    # ── OBV breakout ──────────────────────────────────────────────────────────
    if idx > 0 and pd.notna(row.get("OBV")):
        if row["Close"] > pv("Close") and row["OBV"] > data["OBV_Roll_Max"].iloc[idx-1]:
            sigs.append("📈 OBV突破買入")
        elif row["Close"] < pv("Close") and row["OBV"] < data["OBV_Roll_Min"].iloc[idx-1]:
            sigs.append("📉 OBV突破賣出")

    # ── VIX ───────────────────────────────────────────────────────────────────
    vix_now = row.get("VIX", np.nan)
    if idx > 0 and pd.notna(vix_now):
        vix_prev = data["VIX"].iloc[idx-1]
        if pd.notna(vix_prev):
            if vix_now > p["VIX_HIGH"] and vix_now > vix_prev:  sigs.append("📉 VIX恐慌賣出")
            elif vix_now < p["VIX_LOW"] and vix_now < vix_prev: sigs.append("📈 VIX平靜買入")
        ef = row.get("VIX_EMA_Fast", np.nan)
        es = row.get("VIX_EMA_Slow", np.nan)
        ef_p = data["VIX_EMA_Fast"].iloc[idx-1] if "VIX_EMA_Fast" in data.columns else np.nan
        es_p = data["VIX_EMA_Slow"].iloc[idx-1] if "VIX_EMA_Slow" in data.columns else np.nan
        if pd.notna(ef) and pd.notna(es) and pd.notna(ef_p) and pd.notna(es_p):
            if ef > es and ef_p <= es_p:  sigs.append("📉 VIX上升趨勢賣出")
            elif ef < es and ef_p >= es_p: sigs.append("📈 VIX下降趨勢買入")

    return ", ".join(sigs) if sigs else ""


# ═════════════════════════════════════════════════════════════════════════════
#  COMPREHENSIVE INTERPRETATION
# ═════════════════════════════════════════════════════════════════════════════

def comprehensive_interp(df: pd.DataFrame, dense_areas, VIX_HIGH, VIX_LOW) -> str:
    last5  = df.tail(5)
    bull   = last5["K線形態"].isin(["錘子線","大陽線","看漲吞噬","刺透形態","早晨之星"]).sum()
    bear   = last5["K線形態"].isin(["射擊之星","大陰線","看跌吞噬","烏雲蓋頂","黃昏之星"]).sum()
    hi_vol = (last5["成交量標記"] == "放量").sum()

    # FIX: build dense_desc BEFORE return statements
    dense_desc = ""
    if dense_areas:
        ctrs = [f"{a['price_center']:.2f}" for a in dense_areas]
        dense_desc = f"，成交密集區：{', '.join(ctrs)}"

    vwap_v = last5["VWAP"].iloc[-1]; c_v = last5["Close"].iloc[-1]
    vwap_s = "多頭" if pd.notna(vwap_v) and c_v > vwap_v else "空頭"
    mfi_v  = last5["MFI"].iloc[-1]
    mfi_s  = f"MFI={mfi_v:.0f}({'超賣' if mfi_v<20 else '超買' if mfi_v>80 else '中性'})"
    obv_s  = "OBV↑確認量能" if last5["OBV"].iloc[-1] > last5["OBV"].iloc[0] else "OBV↓警示"
    vix_v  = last5["VIX"].iloc[-1]
    vix_s  = f"VIX={'N/A' if pd.isna(vix_v) else f'{vix_v:.1f}(恐慌)' if vix_v>VIX_HIGH else f'{vix_v:.1f}(平靜)' if vix_v<VIX_LOW else f'{vix_v:.1f}'}"
    suffix = f"｜{vwap_s} VWAP，{mfi_s}，{obv_s}，{vix_s}{dense_desc}"

    if bull >= 3 and hi_vol >= 3:
        return f"多方主導，多根看漲形態放量，強勢上漲趨勢。{suffix}。💡 建議關注買入機會。"
    elif bear >= 3 and hi_vol >= 3:
        return f"空方主導，多根看跌形態放量，強勢下跌趨勢。{suffix}。⚠️ 建議注意賣出風險。"
    elif bull >= 2 and bear >= 2:
        return f"多空激烈爭奪，方向不明。{suffix}。📊 建議觀望。"
    else:
        return f"無明顯趨勢，持續觀察。{suffix}。"


# ═════════════════════════════════════════════════════════════════════════════
#  UI - SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 參數設定")
    input_tickers     = st.text_input("股票代號（逗號分隔）",
                                       "TSLA, NIO, META, GOOGL, AAPL, NVDA, AMZN, MSFT, TSM")
    selected_period   = st.selectbox("時間範圍",
                                      ["1d","5d","1mo","3mo","6mo","1y","2y","5y","ytd","max"], index=5)
    selected_interval = st.selectbox("資料間隔",
                                      ["1m","5m","15m","30m","60m","1h","1d","5d","1wk","1mo"], index=6)
    st.subheader("信號閾值")
    HIGH_N_HIGH_TH   = st.number_input("Close-to-High",     0.1, 1.0, 0.9, 0.1)
    LOW_N_LOW_TH     = st.number_input("Close-to-Low",      0.1, 1.0, 0.9, 0.1)
    PRICE_TH         = st.number_input("股價異動閾值 (%)",   0.1, 200.0, 80.0, 0.1)
    VOLUME_TH        = st.number_input("成交量異動閾值 (%)", 0.1, 200.0, 80.0, 0.1)
    PC_TH            = st.number_input("轉折 Price (%)",     0.1, 200.0,  5.0, 0.1)
    VC_TH            = st.number_input("轉折 Volume (%)",    0.1, 200.0, 10.0, 0.1)
    GAP_TH           = st.number_input("跳空閾值 (%)",       0.1,  50.0,  1.0, 0.1)
    CONT_UP          = st.number_input("連續上漲閾值 (根)",  1, 20, 3, 1)
    CONT_DOWN        = st.number_input("連續下跌閾值 (根)",  1, 20, 3, 1)
    PERCENTILE_TH    = st.selectbox("百分位 (%)",            [1, 5, 10, 20], index=1)
    st.subheader("K 線形態")
    BODY_RATIO_TH    = st.number_input("實體占比",  0.1, 0.9, 0.6, 0.05)
    SHADOW_RATIO_TH  = st.number_input("影線長度",  0.1, 3.0, 2.0, 0.1)
    DOJI_BODY_TH     = st.number_input("十字星閾值",0.01, 0.2, 0.1, 0.01)
    st.subheader("MFI")
    MFI_WIN          = st.number_input("MFI 背離窗口", 3, 20, 5, 1)
    st.subheader("VIX")
    VIX_HIGH_TH      = st.number_input("VIX 恐慌閾值", 20.0, 50.0, 30.0, 1.0)
    VIX_LOW_TH       = st.number_input("VIX 平靜閾值", 10.0, 25.0, 20.0, 1.0)
    VIX_EMA_FAST     = st.number_input("VIX EMA 快", 3, 15,  5, 1)
    VIX_EMA_SLOW     = st.number_input("VIX EMA 慢", 8, 25, 10, 1)
    st.subheader("成交密集區")
    VP_BINS          = st.number_input("分箱數量",  10, 200, 50,  5)
    VP_WINDOW        = st.number_input("K 線根數",  20, 500, 100, 10)
    VP_TOP_N         = st.number_input("顯示前 N",   1,   5,   3,  1)
    VP_SHOW          = st.checkbox("標記密集區", True)
    st.subheader("回測")
    BT_MIN_COMBO     = st.number_input("最少組合數", 2, 3, 2, 1)
    BT_MAX_COMBO     = st.number_input("最多組合數", 2, 5, 3, 1)
    BT_MIN_OCC       = st.number_input("最少次數",   2, 10, 3, 1)
    st.subheader("刷新")
    REFRESH_INTERVAL = st.selectbox("刷新間隔 (秒)", [30, 60, 90, 120, 180, 300], index=4)

# Pack params dict (avoids massive function signatures)
PARAMS = dict(
    HIGH_N_HIGH_TH=HIGH_N_HIGH_TH, LOW_N_LOW_TH=LOW_N_LOW_TH,
    PRICE_TH=PRICE_TH, VOLUME_TH=VOLUME_TH,
    PC_TH=PC_TH, VC_TH=VC_TH, GAP_TH=GAP_TH,
    CONT_UP=CONT_UP, CONT_DOWN=CONT_DOWN,
    MFI_WIN=int(MFI_WIN),
    VIX_HIGH=VIX_HIGH_TH, VIX_LOW=VIX_LOW_TH,
)

selected_tickers = [t.strip().upper() for t in input_tickers.split(",") if t.strip()]

# ── Telegram signal selection ─────────────────────────────────────────────────
ALL_SIGNAL_TYPES = sorted([
    "📈 Low>High","📈 MACD買入","📈 EMA買入","📈 價格趨勢買入","📈 價格趨勢買入(量)",
    "📈 價格趨勢買入(量%)","📈 普通跳空(上)","📈 突破跳空(上)","📈 持續跳空(上)",
    "📈 衰竭跳空(上)","📈 連續向上買入","📈 SMA50上升趨勢","📈 SMA50_200上升趨勢",
    "📈 新买入信号","📈 RSI-MACD Oversold Crossover","📈 EMA-SMA Uptrend Buy",
    "📈 Volume-MACD Buy","📈 EMA10_30買入","📈 EMA10_30_40強烈買入",
    "📈 看漲吞沒","📈 刺透形態","📈 錘頭線","📈 早晨之星",
    "📈 VWAP買入","📈 MFI牛背離買入","📈 OBV突破買入",
    "📈 VIX平靜買入","📈 VIX下降趨勢買入","✅ 量價","🔄 新转折点",
] + list(SELL_SIGNALS))

selected_signals = st.multiselect("選擇需要 Telegram 推播的信號",
                                   ALL_SIGNAL_TYPES, default=["📈 新买入信号"])

# ── Telegram conditions table ─────────────────────────────────────────────────
st.subheader("📋 Telegram 觸發條件配置（可編輯）")
default_conds = pd.DataFrame({
    "排名":     ["1","2","3","4","5"],
    "異動標記": [
        "📈 價格趨勢買入, 📈 持續跳空(上), 📈 SMA50上升趨勢, 📈 OBV突破買入",
        "📈 Low>High, 📈 價格趨勢買入, 📈 SMA50上升趨勢",
        "📈 連續向上買入, 📈 SMA50上升趨勢, 📈 EMA-SMA Uptrend Buy",
        "📈 突破跳空(上), 📈 新买入信号, 📈 EMA-SMA Uptrend Buy",
        "📈 EMA買入, 📈 連續向上買入, 📈 SMA50上升趨勢",
    ],
    "成交量標記": ["放量","縮量","放量","放量","縮量"],
    "K線形態":   ["大陽線","普通K線","大陽線","射擊之星","看漲吞噬"],
})
telegram_conditions = st.data_editor(
    default_conds, num_rows="dynamic",
    column_config={
        "排名":       st.column_config.TextColumn("排名"),
        "異動標記":   st.column_config.TextColumn("異動標記"),
        "成交量標記": st.column_config.SelectboxColumn("成交量標記", options=["放量","縮量"]),
        "K線形態":    st.column_config.TextColumn("K線形態"),
    },
    use_container_width=True,
)

st.title("📊 股票監控儀表板")
st.caption(f"⏱ 更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

tabs = st.tabs([f"📈 {t}" for t in selected_tickers] + ["🔬 回測分析"])

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP  (each ticker)
# ═════════════════════════════════════════════════════════════════════════════

for tab_idx, ticker in enumerate(selected_tickers):
    with tabs[tab_idx]:
        try:
            # ── Fetch data ────────────────────────────────────────────────────
            stock = yf.Ticker(ticker)
            data  = stock.history(period=selected_period, interval=selected_interval).reset_index()
            if data.empty or len(data) < 5:
                st.warning(f"⚠️ {ticker} 數據不足"); continue

            if "Date" in data.columns:
                data = data.rename(columns={"Date": "Datetime"})
            # FIX: strip timezone
            data["Datetime"] = pd.to_datetime(data["Datetime"]).dt.tz_localize(None)

            # ── Basic columns ─────────────────────────────────────────────────
            data["Price Change %"]    = data["Close"].pct_change().round(4) * 100
            data["Volume Change %"]   = data["Volume"].pct_change().round(4) * 100
            hl_range = (data["High"] - data["Low"]).replace(0, np.nan)
            data["Close_N_High"]      = (data["Close"] - data["Low"])   / hl_range
            data["Close_N_Low"]       = (data["High"]  - data["Close"]) / hl_range
            data["前5均量"]            = data["Volume"].rolling(5).mean()
            data["前5均價ABS"]         = data["Price Change %"].abs().rolling(5).mean()
            data["📈 股價漲跌幅(%)"]   = ((data["Price Change %"].abs() - data["前5均價ABS"]) /
                                           data["前5均價ABS"].replace(0, np.nan)).round(4) * 100
            data["📊 成交量變動幅(%)"] = ((data["Volume"] - data["前5均量"]) /
                                           data["前5均量"].replace(0, np.nan)).round(4) * 100

            # ── Indicators ───────────────────────────────────────────────────
            data["MACD"], data["Signal_Line"], data["Histogram"] = calculate_macd(data)
            data["EMA5"]  = data["Close"].ewm(span=5,   adjust=False).mean()
            data["EMA10"] = data["Close"].ewm(span=10,  adjust=False).mean()
            data["EMA30"] = data["Close"].ewm(span=30,  adjust=False).mean()
            data["EMA40"] = data["Close"].ewm(span=40,  adjust=False).mean()
            data["SMA50"] = data["Close"].rolling(50).mean()
            data["SMA200"]= data["Close"].rolling(200).mean()
            data["RSI"]   = calculate_rsi(data)
            data["VWAP"]  = calculate_vwap(data)
            data["MFI"]   = calculate_mfi(data)
            data["OBV"]   = calculate_obv(data)

            # Continuous
            data["Up"]   = (data["Close"] > data["Close"].shift(1)).astype(int)
            data["Down"] = (data["Close"] < data["Close"].shift(1)).astype(int)
            data["Continuous_Up"]   = data["Up"]   * (data["Up"].groupby(   (data["Up"]   == 0).cumsum()).cumcount() + 1)
            data["Continuous_Down"] = data["Down"] * (data["Down"].groupby( (data["Down"] == 0).cumsum()).cumcount() + 1)

            W = int(MFI_WIN)
            data["High_Max"]        = data["High"].rolling(W).max()
            data["Low_Min"]         = data["Low"].rolling(W).min()
            data["Close_Roll_Max"]  = data["Close"].rolling(W).max()
            data["Close_Roll_Min"]  = data["Close"].rolling(W).min()
            data["MFI_Roll_Max"]    = data["MFI"].rolling(W).max()
            data["MFI_Roll_Min"]    = data["MFI"].rolling(W).min()
            data["MFI_Bear_Div"]    = (data["Close"] == data["Close_Roll_Max"]) & (data["MFI"] < data["MFI_Roll_Max"].shift(1))
            data["MFI_Bull_Div"]    = (data["Close"] == data["Close_Roll_Min"]) & (data["MFI"] > data["MFI_Roll_Min"].shift(1))
            data["OBV_Roll_Max"]    = data["OBV"].rolling(20).max()
            data["OBV_Roll_Min"]    = data["OBV"].rolling(20).min()

            # ── VIX ──────────────────────────────────────────────────────────
            vix_df = get_vix_data(selected_period, selected_interval)
            if not vix_df.empty:
                data = data.merge(vix_df, on="Datetime", how="left")
            else:
                data["VIX"] = np.nan; data["VIX_Change_Pct"] = np.nan
            if not data["VIX"].isna().all():
                data["VIX_EMA_Fast"] = data["VIX"].ewm(span=int(VIX_EMA_FAST), adjust=False).mean()
                data["VIX_EMA_Slow"] = data["VIX"].ewm(span=int(VIX_EMA_SLOW), adjust=False).mean()
            else:
                data["VIX_EMA_Fast"] = np.nan; data["VIX_EMA_Slow"] = np.nan

            # ── Signals ───────────────────────────────────────────────────────
            data["異動標記"] = compute_all_signals(data, PARAMS)

            # ── K-line patterns (cached) ──────────────────────────────────────
            _buster = str(round(data["Close"].iloc[-1], 4))
            kdf = get_kline_patterns(ticker, selected_period, selected_interval,
                                     BODY_RATIO_TH, SHADOW_RATIO_TH, DOJI_BODY_TH, _buster)
            kdf["Datetime"] = pd.to_datetime(kdf["Datetime"]).dt.tz_localize(None)
            data = data.merge(kdf, on="Datetime", how="left")
            data["K線形態"]  = data["K線形態"].fillna("普通K線")
            data["單根解讀"] = data["單根解讀"].fillna("波動有限")

            data["成交量標記"] = data.apply(
                lambda r: "放量" if pd.notna(r["前5均量"]) and r["Volume"] > r["前5均量"] else "縮量", axis=1)

            # ── Volume profile ────────────────────────────────────────────────
            dense_areas = calculate_volume_profile(data, int(VP_BINS), int(VP_WINDOW), int(VP_TOP_N))
            latest_close = data["Close"].iloc[-1]
            near_dense = False; near_dense_info = ""
            for a in dense_areas:
                if a["price_low"] <= latest_close <= a["price_high"]:
                    near_dense = True; near_dense_info = f"位於密集區 {a['price_low']:.2f}~{a['price_high']:.2f}"; break
                if abs(latest_close - a["price_center"]) / a["price_center"] * 100 <= 1.0:
                    near_dense = True; near_dense_info = f"接近密集中心 {a['price_center']:.2f}"; break

            # ── Metrics row ───────────────────────────────────────────────────
            try:
                prev_close = stock.info.get("previousClose", data["Close"].iloc[-2])
            except Exception:
                prev_close = data["Close"].iloc[-2]
            cur_price = data["Close"].iloc[-1]
            px_chg = cur_price - prev_close
            px_pct = px_chg / prev_close * 100 if prev_close else 0
            cur_vol = data["Volume"].iloc[-1]; prev_vol = data["Volume"].iloc[-2]
            v_chg = cur_vol - prev_vol; v_pct = v_chg / prev_vol * 100 if prev_vol else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(f"💰 {ticker}", f"${cur_price:.2f}", f"{px_chg:+.2f} ({px_pct:+.2f}%)")
            c2.metric("📊 成交量", f"{cur_vol:,.0f}", f"{v_pct:+.1f}%")
            c3.metric("📈 RSI", f"{data['RSI'].iloc[-1]:.1f}")
            c4.metric("📉 MACD", f"{data['MACD'].iloc[-1]:.3f}")
            vix_v = data["VIX"].iloc[-1]
            c5.metric("⚡ VIX", f"{vix_v:.1f}" if pd.notna(vix_v) else "N/A")

            if near_dense:
                st.info(f"⚠️ {ticker} 靠近成交密集區：{near_dense_info}")

            # ── Comprehensive interpretation ───────────────────────────────────
            st.subheader("📝 綜合解讀")
            st.write(comprehensive_interp(data, dense_areas, VIX_HIGH_TH, VIX_LOW_TH))

            # ── Chart (FIX: 4 rows, no overlaying axes) ────────────────────────
            st.subheader(f"📈 {ticker} K 線技術圖表")
            plot_d = data.tail(60).copy()
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                subplot_titles=("K線 / EMA / VWAP", "成交量 / OBV", "RSI", "MFI"),
                row_heights=[0.45, 0.2, 0.175, 0.175],
                vertical_spacing=0.04,
            )
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=plot_d["Datetime"], open=plot_d["Open"],
                high=plot_d["High"], low=plot_d["Low"], close=plot_d["Close"],
                name="K線", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
            ), row=1, col=1)
            # EMAs / VWAP (FIX: use go.Scatter, not px.line)
            for col_n, clr, w in [("EMA5","#FF6B6B",1.2), ("EMA10","#4ECDC4",1.2),
                                    ("EMA30","#45B7D1",1.2), ("EMA40","#96CEB4",1.2),
                                    ("VWAP","#BB86FC",2.0)]:
                if col_n in plot_d.columns:
                    fig.add_trace(go.Scatter(x=plot_d["Datetime"], y=plot_d[col_n],
                                             mode="lines", name=col_n,
                                             line=dict(color=clr, width=w)), row=1, col=1)
            # Dense areas
            if VP_SHOW and dense_areas and len(plot_d) >= 10:
                x0 = plot_d["Datetime"].iloc[-min(50, len(plot_d))]
                x1 = plot_d["Datetime"].iloc[-1]
                for i, a in enumerate(dense_areas):
                    fig.add_shape(type="rect", x0=x0, x1=x1,
                                  y0=a["price_low"], y1=a["price_high"],
                                  fillcolor="rgba(255,165,0,0.12)", line_width=0,
                                  row=1, col=1)
                    fig.add_hline(y=a["price_center"], line_dash="dot", line_color="orange",
                                  annotation_text=f"密集 {a['price_center']:.2f}",
                                  annotation_position="left" if i%2==0 else "right",
                                  row=1, col=1)
            # Volume bars
            vcols = ["#26a69a" if c>=o else "#ef5350" for c,o in zip(plot_d["Close"], plot_d["Open"])]
            fig.add_trace(go.Bar(x=plot_d["Datetime"], y=plot_d["Volume"],
                                  name="成交量", marker_color=vcols, opacity=0.6), row=2, col=1)
            # OBV (row 2, same panel)
            fig.add_trace(go.Scatter(x=plot_d["Datetime"], y=plot_d["OBV"],
                                      mode="lines", name="OBV",
                                      line=dict(color="#FF8C00", width=1.5),
                                      yaxis="y4"), row=2, col=1)
            # RSI
            fig.add_trace(go.Scatter(x=plot_d["Datetime"], y=plot_d["RSI"],
                                      mode="lines", name="RSI",
                                      line=dict(color="#2196F3", width=1.5)), row=3, col=1)
            for lvl, clr in [(70,"red"),(50,"gray"),(30,"green")]:
                fig.add_hline(y=lvl, line_dash="dash", line_color=clr, line_width=0.7, row=3, col=1)
            # MFI
            fig.add_trace(go.Scatter(x=plot_d["Datetime"], y=plot_d["MFI"],
                                      mode="lines", name="MFI",
                                      line=dict(color="#8B4513", width=1.5)), row=4, col=1)
            for lvl, clr in [(80,"red"),(20,"green")]:
                fig.add_hline(y=lvl, line_dash="dash", line_color=clr, line_width=0.7, row=4, col=1)
            # Signal annotations
            annot_cfg = {
                "📈 新买入信号":  ("▲","#2ecc71","bottom center",1),
                "📉 新卖出信号":  ("▼","#e74c3c","top center",   1),
                "📈 VWAP買入":   ("V↑","#BB86FC","bottom center",1),
                "📉 VWAP賣出":   ("V↓","#BB86FC","top center",   1),
                "📈 OBV突破買入":("O↑","#FF8C00","bottom center",2),
                "📉 OBV突破賣出":("O↓","#FF8C00","top center",   2),
                "📈 MFI牛背離買入":("M↑","#8B4513","bottom center",4),
                "📉 MFI熊背離賣出":("M↓","#8B4513","top center",  4),
                "📈 MACD買入":   ("MC↑","#4ECDC4","bottom center",1),
                "📉 MACD賣出":   ("MC↓","#FF6B6B","top center",   1),
            }
            for i in range(1, len(plot_d)):
                marks = str(plot_d["異動標記"].iloc[i])
                dt, cl = plot_d["Datetime"].iloc[i], plot_d["Close"].iloc[i]
                for sig, (sym, clr, pos, row_n) in annot_cfg.items():
                    if sig in marks:
                        fig.add_trace(go.Scatter(
                            x=[dt], y=[cl], mode="markers+text",
                            marker=dict(symbol="circle", size=9, color=clr),
                            text=[sym], textposition=pos,
                            showlegend=False,
                        ), row=row_n, col=1)

            fig.update_layout(
                height=920, template="plotly_dark", showlegend=True,
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=-0.04, font=dict(size=11)),
                margin=dict(l=40, r=40, t=60, b=40),
            )
            fig.update_yaxes(title_text="價格",  row=1, col=1)
            fig.update_yaxes(title_text="成交量", row=2, col=1)
            fig.update_yaxes(title_text="RSI",   row=3, col=1, range=[0,100])
            fig.update_yaxes(title_text="MFI",   row=4, col=1, range=[0,100])
            st.plotly_chart(fig, use_container_width=True,
                            key=f"chart_{ticker}_{datetime.now().strftime('%H%M%S')}")

            # ── Signal success rate ────────────────────────────────────────────
            st.subheader(f"📊 {ticker} 各信號勝率")
            data["_next_up"]  = data["Close"].shift(-1) > data["Close"]
            data["_next_dn"]  = data["Close"].shift(-1) < data["Close"]
            sr_rows = []
            all_sigs_found = set()
            for marks in data["異動標記"].dropna():
                for s in str(marks).split(", "):
                    if s.strip():
                        all_sigs_found.add(s.strip())
            for sig in sorted(all_sigs_found):
                sub = data[data["異動標記"].str.contains(sig, na=False, regex=False)]
                n   = len(sub)
                if n == 0:
                    continue
                if sig in SELL_SIGNALS:
                    ok = sub["_next_dn"].sum(); dir_ = "做空"
                else:
                    ok = sub["_next_up"].sum(); dir_ = "做多"
                wr = ok / n * 100
                sr_rows.append({"信號":sig,"方向":dir_,"勝率(%)":f"{wr:.1f}%","次數":n})
            if sr_rows:
                sr_df = pd.DataFrame(sr_rows).sort_values("勝率(%)", ascending=False)
                st.dataframe(sr_df, use_container_width=True,
                             column_config={"信號": st.column_config.TextColumn(width="large")})

            # ── History table ─────────────────────────────────────────────────
            st.subheader(f"📋 {ticker} 歷史資料（最近 20 筆）")
            show_cols = [c for c in ["Datetime","Open","Low","High","Close","Volume",
                                      "Price Change %","Volume Change %","MACD","RSI",
                                      "VWAP","MFI","OBV","VIX",
                                      "異動標記","成交量標記","K線形態","單根解讀"] if c in data.columns]
            st.dataframe(data[show_cols].tail(20), height=460, use_container_width=True,
                         column_config={"異動標記":   st.column_config.TextColumn(width="large"),
                                        "單根解讀": st.column_config.TextColumn(width="large")})

            # ── Percentile table ───────────────────────────────────────────────
            with st.expander(f"📊 前 {PERCENTILE_TH}% 數據範圍"):
                rng_rows = []
                for cn in ["Price Change %","Volume Change %","Volume","📈 股價漲跌幅(%)","📊 成交量變動幅(%)"]:
                    if cn not in data.columns: continue
                    s = data[cn].dropna().sort_values(ascending=False)
                    n = max(1, int(len(s) * PERCENTILE_TH / 100))
                    rng_rows += [
                        {"指標":cn,"範圍":"Top",    "最大":f"{s.head(n).max():.2f}","最小":f"{s.head(n).min():.2f}"},
                        {"指標":cn,"範圍":"Bottom", "最大":f"{s.tail(n).max():.2f}","最小":f"{s.tail(n).min():.2f}"},
                    ]
                if rng_rows:
                    st.dataframe(pd.DataFrame(rng_rows), use_container_width=True)

            # ── CSV download ──────────────────────────────────────────────────
            st.download_button(
                label=f"📥 下載 {ticker} CSV",
                data=data.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            # ── Telegram / Email alerts ────────────────────────────────────────
            K_str  = str(data["異動標記"].iloc[-1])
            K_list = [s.strip() for s in K_str.split(", ") if s.strip()]

            # Selected signal push
            for sig in selected_signals:
                if sig in K_list:
                    send_telegram_alert(
                        f"📡 {ticker} 信號「{sig}」"
                        f" @ ${cur_price:.2f} | RSI={data['RSI'].iloc[-1]:.1f}")

            # FIX: matched_rank initialised to None BEFORE condition checks
            matched_rank = None
            for _, cond_row in telegram_conditions.iterrows():
                req = [s.strip() for s in str(cond_row["異動標記"]).split(", ") if s.strip()]
                if (all(s in K_list for s in req) and
                        data["成交量標記"].iloc[-1] == cond_row["成交量標記"] and
                        data["K線形態"].iloc[-1]  == cond_row["K線形態"]):
                    matched_rank = cond_row["排名"]
                    break

            if matched_rank is not None:
                send_telegram_alert(
                    f"🟢 趨勢反轉買入 {ticker}:{selected_interval} "
                    f"${cur_price:.2f} | {K_str[:120]} | "
                    f"{data['成交量標記'].iloc[-1]} | {data['K線形態'].iloc[-1]} | 排名:{matched_rank}")

            # Breakout/breakdown alerts
            if pd.notna(data["High_Max"].iloc[-1]) and data["High"].iloc[-1] >= data["High_Max"].iloc[-1]:
                send_telegram_alert(f"🚀 {ticker} 破 {W}K 新高 ${data['High'].iloc[-1]:.2f}")
            if pd.notna(data["Low_Min"].iloc[-1]) and data["Low"].iloc[-1] <= data["Low_Min"].iloc[-1]:
                send_telegram_alert(f"🔻 {ticker} 穿 {W}K 新低 ${data['Low'].iloc[-1]:.2f}")

            # Email (consolidated)
            sig_dict = {
                "macd_buy":       "📈 MACD買入"  in K_list,
                "macd_sell":      "📉 MACD賣出"  in K_list,
                "new_buy":        "📈 新买入信号" in K_list,
                "new_sell":       "📉 新卖出信号" in K_list,
                "vwap_buy":       "📈 VWAP買入"  in K_list,
                "vwap_sell":      "📉 VWAP賣出"  in K_list,
                "obv_buy":        "📈 OBV突破買入" in K_list,
                "obv_sell":       "📉 OBV突破賣出" in K_list,
                "mfi_bull":       "📈 MFI牛背離買入" in K_list,
                "mfi_bear":       "📉 MFI熊背離賣出" in K_list,
                "vix_panic":      "📉 VIX恐慌賣出" in K_list,
                "vix_calm":       "📈 VIX平靜買入" in K_list,
                "bullish_eng":    "📈 看漲吞沒"   in K_list,
                "bearish_eng":    "📉 看跌吞沒"   in K_list,
                "morning_star":   "📈 早晨之星"   in K_list,
                "evening_star":   "📉 黃昏之星"   in K_list,
                "hammer":         "📈 錘頭線"     in K_list,
                "hanging_man":    "📉 上吊線"     in K_list,
            }
            if any(sig_dict.values()):
                send_email_alert(ticker, px_pct, v_pct, sig_dict)

        except Exception as e:
            st.error(f"⚠️ {ticker} 發生錯誤：{e}")
            with st.expander("詳細錯誤"):
                st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
#  BACKTEST TAB
# ═════════════════════════════════════════════════════════════════════════════

with tabs[-1]:
    st.header("🔬 回測：組合信號勝率分析")
    st.info("""
    **分析哪些技術指標同時出現時勝率最高**
    - 做多勝率 = 信號出現後下一根 K 線收漲的比例
    - 做空勝率 = 信號出現後下一根 K 線收跌的比例
    - ⚠️ 回測僅供參考，請結合風險管理進行決策
    """)

    bt_ticker = st.selectbox("選擇回測股票", selected_tickers, key="bt_ticker")
    col_a, col_b, col_c = st.columns(3)
    bt_min = col_a.number_input("最少組合數", 2, 3, int(BT_MIN_COMBO), 1, key="bt_min")
    bt_max = col_b.number_input("最多組合數", 2, 5, int(BT_MAX_COMBO), 1, key="bt_max")
    bt_occ = col_c.number_input("最少出現次數", 2, 20, int(BT_MIN_OCC), 1, key="bt_occ")

    if st.button("🚀 開始回測", type="primary"):
        with st.spinner(f"正在計算 {bt_ticker} 信號組合勝率..."):
            try:
                bt_data = yf.Ticker(bt_ticker).history(
                    period=selected_period, interval=selected_interval).reset_index()
                if "Date" in bt_data.columns:
                    bt_data = bt_data.rename(columns={"Date": "Datetime"})
                bt_data["Datetime"] = pd.to_datetime(bt_data["Datetime"]).dt.tz_localize(None)
                bt_data["前5均量"]   = bt_data["Volume"].rolling(5).mean()
                bt_data["Price Change %"]  = bt_data["Close"].pct_change() * 100
                bt_data["Volume Change %"] = bt_data["Volume"].pct_change() * 100
                bt_data["MACD"], bt_data["Signal_Line"], _ = calculate_macd(bt_data)
                bt_data["RSI"]    = calculate_rsi(bt_data)
                for span, name in [(5,"EMA5"),(10,"EMA10"),(30,"EMA30"),(40,"EMA40")]:
                    bt_data[name] = bt_data["Close"].ewm(span=span, adjust=False).mean()
                bt_data["SMA50"]  = bt_data["Close"].rolling(50).mean()
                bt_data["SMA200"] = bt_data["Close"].rolling(200).mean()
                bt_data["VWAP"]   = calculate_vwap(bt_data)
                bt_data["MFI"]    = calculate_mfi(bt_data)
                bt_data["OBV"]    = calculate_obv(bt_data)
                bt_data["Up"]   = (bt_data["Close"] > bt_data["Close"].shift(1)).astype(int)
                bt_data["Down"] = (bt_data["Close"] < bt_data["Close"].shift(1)).astype(int)
                bt_data["Continuous_Up"]   = bt_data["Up"]   * (bt_data["Up"].groupby(   (bt_data["Up"]   == 0).cumsum()).cumcount() + 1)
                bt_data["Continuous_Down"] = bt_data["Down"] * (bt_data["Down"].groupby( (bt_data["Down"] == 0).cumsum()).cumcount() + 1)
                W2 = int(MFI_WIN)
                bt_data["High_Max"]       = bt_data["High"].rolling(W2).max()
                bt_data["Low_Min"]        = bt_data["Low"].rolling(W2).min()
                bt_data["Close_Roll_Max"] = bt_data["Close"].rolling(W2).max()
                bt_data["Close_Roll_Min"] = bt_data["Close"].rolling(W2).min()
                bt_data["MFI_Roll_Max"]   = bt_data["MFI"].rolling(W2).max()
                bt_data["MFI_Roll_Min"]   = bt_data["MFI"].rolling(W2).min()
                bt_data["MFI_Bear_Div"]   = (bt_data["Close"] == bt_data["Close_Roll_Max"]) & (bt_data["MFI"] < bt_data["MFI_Roll_Max"].shift(1))
                bt_data["MFI_Bull_Div"]   = (bt_data["Close"] == bt_data["Close_Roll_Min"]) & (bt_data["MFI"] > bt_data["MFI_Roll_Min"].shift(1))
                bt_data["OBV_Roll_Max"]   = bt_data["OBV"].rolling(20).max()
                bt_data["OBV_Roll_Min"]   = bt_data["OBV"].rolling(20).min()
                bt_data["VIX"] = np.nan; bt_data["VIX_EMA_Fast"] = np.nan; bt_data["VIX_EMA_Slow"] = np.nan
                bt_data["📈 股價漲跌幅(%)"]   = np.nan
                bt_data["📊 成交量變動幅(%)"] = np.nan
                bt_data["Close_N_High"] = np.nan; bt_data["Close_N_Low"] = np.nan

                data = bt_data  # needed for closure in compute_all_signals
                bt_data["異動標記"] = compute_all_signals(bt_data, PARAMS)

                combo_df = backtest_signal_combinations(
                    bt_data, min_combo=int(bt_min), max_combo=int(bt_max), min_occ=int(bt_occ))

                if combo_df.empty:
                    st.warning("資料不足。請選擇更長的時間範圍（建議 1y 以上）。")
                else:
                    total_combos = len(combo_df)
                    high_wr_df   = combo_df[combo_df["勝率(%)"] >= 60]
                    st.success(f"✅ 共找到 {total_combos} 個有效組合，其中 {len(high_wr_df)} 個勝率 ≥ 60%")

                    if not high_wr_df.empty:
                        st.subheader("🏆 高勝率組合（≥ 60%）")
                        st.dataframe(
                            high_wr_df.style.background_gradient(subset=["勝率(%)"], cmap="Greens"),
                            use_container_width=True)

                    st.subheader("📊 全部組合排名")
                    st.dataframe(
                        combo_df.style.background_gradient(subset=["勝率(%)"], cmap="RdYlGn"),
                        use_container_width=True)

                    # Bar chart
                    top15 = combo_df.head(15)
                    bar_colors = ["#2ecc71" if d=="做多" else "#e74c3c" for d in top15["方向"]]
                    fig_bt = go.Figure(go.Bar(
                        x=top15["勝率(%)"],
                        y=top15["信號組合"],
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:.1f}% ({n}次)" for v,n in zip(top15["勝率(%)"], top15["出現次數"])],
                        textposition="outside",
                    ))
                    fig_bt.add_vline(x=60, line_dash="dash", line_color="gold",
                                     annotation_text="60% 基準線")
                    fig_bt.update_layout(
                        title=f"{bt_ticker} 信號組合勝率排名（前 15）",
                        xaxis_title="勝率 (%)", height=600,
                        template="plotly_dark",
                        margin=dict(l=350, r=80, t=60, b=40),
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)

                    # Advice
                    st.subheader("💡 交易建議")
                    if not high_wr_df.empty:
                        best = high_wr_df.iloc[0]
                        st.success(
                            f"**最佳組合**：{best['信號組合']}\n\n"
                            f"- 勝率：**{best['勝率(%)']}%**  |  出現次數：**{best['出現次數']}**  |  方向：**{best['方向']}**\n\n"
                            "⚠️ 回測基於歷史數據，未來不保證相同表現。請結合基本面分析與嚴格止損策略。")
                    else:
                        st.info("目前無 ≥60% 勝率組合，建議延長時間範圍至 1y 以上，或調整信號參數。")

            except Exception as e:
                st.error(f"回測失敗：{e}")
                with st.expander("詳細錯誤"):
                    st.code(traceback.format_exc())

# ═════════════════════════════════════════════════════════════════════════════
#  AUTO REFRESH (FIX: replace while True + time.sleep with time.sleep + st.rerun)
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
col_l, col_r = st.columns([4, 1])
with col_l:
    st.info(f"📡 頁面將在 **{REFRESH_INTERVAL}** 秒後自動刷新")
with col_r:
    if st.button("🔄 立即刷新"):
        st.rerun()

time.sleep(REFRESH_INTERVAL)
st.rerun()
