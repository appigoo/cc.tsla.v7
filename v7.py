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
import json

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

def _tg_escape(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    # Characters that must be escaped in MarkdownV2 plain text
    specials = r"\_*[]()~`>#+-=|{}.!"
    for ch in specials:
        text = text.replace(ch, f"\\{ch}")
    return text


def send_telegram_alert(msg: str) -> tuple:
    """
    Send a plain-text message via Telegram Bot API.
    Returns (success: bool, error_msg: str).
    Uses POST + plain text (no parse_mode) to avoid escaping issues.
    """
    if not (BOT_TOKEN and CHAT_ID):
        return False, "Telegram 未設定（請在 secrets.toml 中設定 BOT_TOKEN 和 CHAT_ID）"
    try:
        url     = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id":                  CHAT_ID,
            "text":                     msg,
            "disable_web_page_preview": True,
            "disable_notification":     False,
        }
        r = requests.post(url, json=payload, timeout=15)
        resp = r.json()
        if r.status_code == 200 and resp.get("ok"):
            return True, ""
        else:
            err = resp.get("description", f"HTTP {r.status_code}")
            return False, f"Telegram API 錯誤：{err}"
    except requests.exceptions.Timeout:
        return False, "Telegram 發送逾時（15 秒）"
    except Exception as e:
        return False, f"Telegram 發送例外：{e}"


def _fmt_vol(vol) -> str:
    """Format volume as readable string: 1,234,567 → 1.23M"""
    try:
        v = float(vol)
        if v >= 1_000_000:
            return f"{v/1_000_000:.2f}M"
        elif v >= 1_000:
            return f"{v/1_000:.1f}K"
        else:
            return f"{v:.0f}"
    except Exception:
        return str(vol)


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

def _calc_wr(sub_next_up: "np.ndarray", is_sell: bool) -> float:
    """
    Helper: compute win rate from a boolean array of next-bar direction.
    Accepts either a pd.Series or np.ndarray for sub["_next_up"].
    """
    n = len(sub_next_up)
    if n == 0:
        return 0.0
    mean_up = float(np.asarray(sub_next_up, dtype=float).mean())
    return (1 - mean_up) * 100 if is_sell else mean_up * 100


def _calc_avg_pnl(close_arr: "np.ndarray", next_close_arr: "np.ndarray",
                  mask: "np.ndarray", is_sell: bool) -> float:
    """
    計算命中該組合的所有K線的平均每筆盈虧(%)。
    包含勝筆和敗筆，反映真實期望報酬。
      做多：pnl = (next_close - close) / close * 100
      做空：pnl = (close - next_close) / close * 100
    """
    c  = close_arr[mask]
    nc = next_close_arr[mask]
    if len(c) == 0:
        return 0.0
    safe_c = np.where(c == 0, np.nan, c)
    pnl    = (nc - c) / safe_c * 100
    if is_sell:
        pnl = -pnl
    valid = pnl[~np.isnan(pnl)]
    return round(float(valid.mean()), 2) if len(valid) > 0 else 0.0


def _build_onehot(df: "pd.DataFrame") -> "tuple[list, dict, np.ndarray]":
    """
    共用前置步驟：
    1. 解析每行異動標記 → signal_sets (list of set)
    2. 收集所有不重複信號 → all_s (sorted list)
    3. 建立 one-hot 布林矩陣 onehot (n_rows × n_signals, dtype=bool)

    one-hot[i, j] = True 表示第 i 根K線包含第 j 個信號。

    向量化篩選：
        mask = onehot[:, [col_A, col_B]].all(axis=1)
    等價於原本的：
        mask = df["_sigs"].apply(lambda s: {A, B}.issubset(s))
    但速度快 10～50 倍（純 NumPy，無 Python 逐行循環）。
    """
    signal_sets = []
    for marks in df["異動標記"].fillna(""):
        sigs = {s.strip() for s in str(marks).split(", ")
                if s.strip() and "🔥" not in s}
        signal_sets.append(sigs)

    all_s     = sorted({s for ss in signal_sets for s in ss})
    sig_index = {s: i for i, s in enumerate(all_s)}
    n_sigs    = len(all_s)
    n_rows    = len(df)

    onehot = np.zeros((n_rows, n_sigs), dtype=bool)
    for row_i, sset in enumerate(signal_sets):
        for s in sset:
            if s in sig_index:
                onehot[row_i, sig_index[s]] = True

    return all_s, sig_index, onehot


def _combo_mask(combo: tuple, sig_index: dict,
                onehot: "np.ndarray") -> "np.ndarray":
    """
    給定一個信號組合 tuple，回傳 shape=(n_rows,) 的布林陣列。
    True 表示該行K線包含組合中的所有信號。
    完全等價於：df["_sigs"].apply(lambda s: set(combo).issubset(s))
    """
    cols = [sig_index[s] for s in combo if s in sig_index]
    if not cols:
        return np.zeros(onehot.shape[0], dtype=bool)
    # np.ndarray[:, cols].all(axis=1) — 純 NumPy，無 Python 逐行循環
    return onehot[:, cols].all(axis=1)


def _base_signal_combos(df: "pd.DataFrame", min_combo: int, max_combo: int,
                         min_occ: int) -> "pd.DataFrame":
    """
    維度 1：純信號組合勝率（向量化加速版）
    回傳欄位：維度, 信號組合, 成交量標記, K線形態, 信號數量, 勝率(%), 平均盈虧(%), 出現次數, 方向
    """
    df = df.copy()
    close_arr      = df["Close"].to_numpy()
    next_close_arr = df["Close"].shift(-1).to_numpy()
    next_up        = (next_close_arr > close_arr)

    all_s, sig_index, onehot = _build_onehot(df)

    rows = []
    for r in range(min_combo, min(max_combo + 1, len(all_s) + 1)):
        for combo in combinations(all_s, r):
            mask    = _combo_mask(combo, sig_index, onehot)
            n_hit   = int(mask.sum())
            if n_hit < min_occ:
                continue
            is_sell = sum(1 for s in combo if s in SELL_SIGNALS) > len(combo) / 2
            rows.append({
                "維度":        "信號組合",
                "信號組合":    " + ".join(combo),
                "成交量標記":  "—",
                "K線形態":     "—",
                "信號數量":    r,
                "勝率(%)":     round(_calc_wr(next_up[mask], is_sell), 1),
                "平均盈虧(%)": _calc_avg_pnl(close_arr, next_close_arr, mask, is_sell),
                "出現次數":    n_hit,
                "方向":        "做空" if is_sell else "做多",
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("勝率(%)", ascending=False).head(30)


def _signal_x_volume_combos(df: "pd.DataFrame", min_combo: int, max_combo: int,
                              min_occ: int) -> "pd.DataFrame":
    """
    維度 2：信號組合 × 成交量標記（向量化加速版）
    """
    if "成交量標記" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    close_arr      = df["Close"].to_numpy()
    next_close_arr = df["Close"].shift(-1).to_numpy()
    next_up        = (next_close_arr > close_arr)
    vol_arr  = df["成交量標記"].to_numpy()
    vol_放量 = (vol_arr == "放量")
    vol_縮量 = (vol_arr == "縮量")

    all_s, sig_index, onehot = _build_onehot(df)

    rows = []
    for r in range(min_combo, min(max_combo + 1, len(all_s) + 1)):
        for combo in combinations(all_s, r):
            base_mask = _combo_mask(combo, sig_index, onehot)
            n_base    = int(base_mask.sum())
            if n_base < min_occ:
                continue
            is_sell = sum(1 for s in combo if s in SELL_SIGNALS) > len(combo) / 2
            for vol_label, vol_mask in [("放量", vol_放量), ("縮量", vol_縮量)]:
                mask  = base_mask & vol_mask
                n_hit = int(mask.sum())
                if n_hit < min_occ:
                    continue
                rows.append({
                    "維度":        "信號+成交量",
                    "信號組合":    " + ".join(combo),
                    "成交量標記":  vol_label,
                    "K線形態":     "—",
                    "信號數量":    r,
                    "勝率(%)":     round(_calc_wr(next_up[mask], is_sell), 1),
                    "平均盈虧(%)": _calc_avg_pnl(close_arr, next_close_arr, mask, is_sell),
                    "出現次數":    n_hit,
                    "方向":        "做空" if is_sell else "做多",
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("勝率(%)", ascending=False).head(30)


def _signal_x_kline_combos(df: "pd.DataFrame", min_combo: int, max_combo: int,
                             min_occ: int) -> "pd.DataFrame":
    """
    維度 3：信號組合 × K線形態（向量化加速版）
    """
    if "K線形態" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    close_arr      = df["Close"].to_numpy()
    next_close_arr = df["Close"].shift(-1).to_numpy()
    next_up        = (next_close_arr > close_arr)
    kline_arr  = df["K線形態"].fillna("普通K線").to_numpy()
    kline_vals = [k for k in df["K線形態"].dropna().unique()
                  if k and k != "普通K線"]

    # 預建每種K線形態的布林遮罩，避免在內層迴圈重複比較
    kline_masks = {kl: (kline_arr == kl) for kl in kline_vals}

    all_s, sig_index, onehot = _build_onehot(df)

    rows = []
    for r in range(min_combo, min(max_combo + 1, len(all_s) + 1)):
        for combo in combinations(all_s, r):
            base_mask = _combo_mask(combo, sig_index, onehot)
            n_base    = int(base_mask.sum())
            if n_base < min_occ:
                continue
            is_sell = sum(1 for s in combo if s in SELL_SIGNALS) > len(combo) / 2
            for kl, kl_mask in kline_masks.items():
                mask  = base_mask & kl_mask
                n_hit = int(mask.sum())
                if n_hit < min_occ:
                    continue
                rows.append({
                    "維度":        "信號+K線形態",
                    "信號組合":    " + ".join(combo),
                    "成交量標記":  "—",
                    "K線形態":     kl,
                    "信號數量":    r,
                    "勝率(%)":     round(_calc_wr(next_up[mask], is_sell), 1),
                    "平均盈虧(%)": _calc_avg_pnl(close_arr, next_close_arr, mask, is_sell),
                    "出現次數":    n_hit,
                    "方向":        "做空" if is_sell else "做多",
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("勝率(%)", ascending=False).head(30)



def _detailed_backtest(df: "pd.DataFrame",
                       signal_combo: str,
                       vol_filter: str = "—",
                       kline_filter: str = "—",
                       direction: str = "做多",
                       hold_bars: int = 1) -> "pd.DataFrame":
    """
    對單一信號組合逐筆展開所有交易記錄，計算每筆盈虧。

    Parameters
    ----------
    df            : 已含異動標記、K線形態、成交量標記的完整回測資料
    signal_combo  : " + " 分隔的信號組合字串
    vol_filter    : "放量" / "縮量" / "—"（不限）
    kline_filter  : K線形態字串 / "—"（不限）
    direction     : "做多" or "做空"
    hold_bars     : 持倉根數（預設 1 = 下一根出場）

    Returns
    -------
    pd.DataFrame：每一筆交易的詳細記錄
    """
    if df.empty or "異動標記" not in df.columns:
        return pd.DataFrame()

    df = df.copy().reset_index(drop=True)

    # 解析組合信號
    required_sigs = [s.strip() for s in signal_combo.split(" + ") if s.strip()]

    # 篩選命中行
    def _hit(row):
        marks = str(row.get("異動標記", ""))
        sigs  = {s.strip() for s in marks.split(", ") if s.strip()}
        sig_ok = all(s in sigs for s in required_sigs)
        vol_ok = (vol_filter in ("—", "全部") or
                  str(row.get("成交量標記","")) == vol_filter)
        kl_ok  = (kline_filter in ("—", "全部") or
                  str(row.get("K線形態","")) == kline_filter)
        return sig_ok and vol_ok and kl_ok

    hit_mask = df.apply(_hit, axis=1)
    hit_idx  = df.index[hit_mask].tolist()

    records = []
    for i in hit_idx:
        exit_i = i + hold_bars
        if exit_i >= len(df):
            continue  # 無出場K線，跳過最後幾筆

        entry_bar  = df.iloc[i]
        exit_bar   = df.iloc[exit_i]

        # 進場價 = 當根收盤（訊號出現時）
        # 出場價 = hold_bars 根後的收盤
        entry_price = float(entry_bar["Close"])
        exit_price  = float(exit_bar["Close"])

        if entry_price <= 0:
            continue

        if direction == "做多":
            pnl_pct  = (exit_price - entry_price) / entry_price * 100
            win      = exit_price > entry_price
        else:  # 做空
            pnl_pct  = (entry_price - exit_price) / entry_price * 100
            win      = exit_price < entry_price

        # RSI / MACD at entry
        entry_rsi  = round(float(entry_bar.get("RSI",  float("nan"))), 1)
        entry_macd = round(float(entry_bar.get("MACD", float("nan"))), 4)
        entry_vol  = entry_bar.get("成交量標記", "—")
        entry_kl   = entry_bar.get("K線形態",    "—")

        # High / Low in the holding period for risk metrics
        hold_slice      = df.iloc[i+1 : exit_i+1]
        max_high        = float(hold_slice["High"].max()) if not hold_slice.empty else exit_price
        min_low         = float(hold_slice["Low"].min())  if not hold_slice.empty else exit_price

        if direction == "做多":
            max_runup_pct  = (max_high - entry_price) / entry_price * 100
            max_drawdown_pct = (entry_price - min_low) / entry_price * 100
        else:
            max_runup_pct  = (entry_price - min_low)  / entry_price * 100
            max_drawdown_pct = (max_high - entry_price) / entry_price * 100

        entry_dt = entry_bar.get("Datetime", "")
        exit_dt  = exit_bar.get("Datetime",  "")

        records.append({
            "序號":        len(records) + 1,
            "進場時間":    str(entry_dt)[:19],
            "出場時間":    str(exit_dt)[:19],
            "持倉根數":    hold_bars,
            "進場價":      round(entry_price, 4),
            "出場價":      round(exit_price,  4),
            "方向":        direction,
            "盈虧(%)":     round(pnl_pct, 2),
            "勝負":        "✅ 勝" if win else "❌ 敗",
            "最大順勢(%)": round(max_runup_pct,   2),
            "最大逆勢(%)": round(max_drawdown_pct, 2),
            "進場RSI":     entry_rsi,
            "進場MACD":    entry_macd,
            "成交量標記":  str(entry_vol),
            "K線形態":     str(entry_kl),
            "觸發信號":    str(entry_bar.get("異動標記", ""))[:120],
        })

    if not records:
        return pd.DataFrame()

    detail_df = pd.DataFrame(records)

    # 累計盈虧（等額投入，每次 100%，簡易模擬）
    detail_df["累計盈虧(%)"] = detail_df["盈虧(%)"].cumsum().round(2)

    # 連勝 / 連敗統計
    wins = (detail_df["勝負"] == "✅ 勝").astype(int)
    loss = (1 - wins)
    streak_win = wins  * (wins .groupby((wins  == 0).cumsum()).cumcount() + 1)
    streak_loss= loss  * (loss.groupby((loss == 0).cumsum()).cumcount() + 1)
    detail_df["連勝數"] = streak_win
    detail_df["連敗數"] = streak_loss

    return detail_df


def _summary_stats(detail_df: "pd.DataFrame") -> dict:
    """
    從逐筆交易記錄計算統計摘要，用於驗證勝率準確度。
    """
    if detail_df.empty:
        return {}

    total    = len(detail_df)
    wins     = (detail_df["勝負"] == "✅ 勝").sum()
    losses   = total - wins
    wr       = wins / total * 100 if total else 0

    pnl      = detail_df["盈虧(%)"]
    win_pnl  = pnl[detail_df["勝負"] == "✅ 勝"]
    loss_pnl = pnl[detail_df["勝負"] == "❌ 敗"]

    avg_win  = win_pnl.mean()  if len(win_pnl)  else 0
    avg_loss = loss_pnl.mean() if len(loss_pnl) else 0

    # Profit factor = total profit / abs(total loss)
    total_profit = win_pnl.sum()   if len(win_pnl)  else 0
    total_loss   = abs(loss_pnl.sum()) if len(loss_pnl) else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # Max drawdown on cumulative PnL series
    cum = detail_df["累計盈虧(%)"]
    roll_max = cum.cummax()
    drawdown = cum - roll_max
    max_dd   = drawdown.min()

    # Expectancy per trade = WR*avg_win - (1-WR)*avg_loss
    expectancy = (wr/100) * avg_win + ((1 - wr/100)) * avg_loss

    # Calmar-like: total PnL / abs(max drawdown)
    calmar = (cum.iloc[-1] / abs(max_dd)) if max_dd != 0 else float("inf")

    max_streak_win  = detail_df["連勝數"].max()
    max_streak_loss = detail_df["連敗數"].max()

    return {
        "總交易筆數":      total,
        "勝出筆數":        int(wins),
        "敗出筆數":        int(losses),
        "實際勝率(%)":     round(wr, 1),
        "平均盈利(%)":     round(avg_win,  2),
        "平均虧損(%)":     round(avg_loss, 2),
        "盈虧比":          round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else "∞",
        "獲利因子":        round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "期望值每筆(%)":   round(expectancy, 2),
        "累計盈虧(%)":     round(float(cum.iloc[-1]), 2),
        "最大回撤(%)":     round(float(max_dd), 2),
        "Calmar比率":      round(calmar, 2) if calmar != float("inf") else "∞",
        "最大連勝":        int(max_streak_win),
        "最大連敗":        int(max_streak_loss),
    }

def backtest_signal_combinations(df: "pd.DataFrame", min_combo=2,
                                  max_combo=4, min_occ=3) -> "pd.DataFrame":
    """保留舊介面相容：只跑維度1（純信號組合）。回測 Tab 直接呼叫三個子函數。"""
    return _base_signal_combos(df, min_combo, max_combo, min_occ)


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

selected_tickers = [t.strip().upper() for t in (input_tickers or "").split(",") if t.strip()]

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

# ── Telegram 發送開關 ────────────────────────────────────────────────────────
if "tg_enabled" not in st.session_state:
    st.session_state["tg_enabled"] = True   # 預設開啟

_sw_col, _sw_info = st.columns([1, 5])
with _sw_col:
    _sw_label = (
        "🟢 Telegram：開啟" if st.session_state["tg_enabled"]
        else "🔴 Telegram：已關閉"
    )
    if st.button(_sw_label, key="tg_toggle_btn", use_container_width=True):
        st.session_state["tg_enabled"] = not st.session_state["tg_enabled"]
        st.rerun()
with _sw_info:
    if st.session_state["tg_enabled"]:
        st.success("🟢 **Telegram 發送開啟**：條件匹配時自動推送訊號", icon="✅")
    else:
        st.warning(
            "🔴 **Telegram 已關閉（調參模式）**：條件匹配時只顯示 UI 提示，不發送任何訊息。"
            "完成調參後請重新開啟。",
            icon="🔕",
        )


selected_signals = st.multiselect("選擇需要 Telegram 推播的信號",
                                   ALL_SIGNAL_TYPES, default=["📈 新买入信号"])

# ── Telegram conditions table ─────────────────────────────────────────────────
st.subheader("📋 Telegram 觸發條件配置（可編輯）")

_TG_DEFAULT = pd.DataFrame({
    "排名":       ["1","2","3","4","5"],
    "異動標記":   [
        "📈 價格趨勢買入, 📈 持續跳空(上), 📈 SMA50上升趨勢, 📈 OBV突破買入",
        "📈 Low>High, 📈 價格趨勢買入, 📈 SMA50上升趨勢",
        "📈 連續向上買入, 📈 SMA50上升趨勢, 📈 EMA-SMA Uptrend Buy",
        "📈 突破跳空(上), 📈 新买入信号, 📈 EMA-SMA Uptrend Buy",
        "📈 EMA買入, 📈 連續向上買入, 📈 SMA50上升趨勢",
    ],
    "成交量標記": ["放量","縮量","放量","放量","縮量"],
    "K線形態":    ["大陽線","普通K線","大陽線","射擊之星","看漲吞噬"],
    "回測勝率":   ["N/A","N/A","N/A","N/A","N/A"],
    "方向":       ["做多","做多","做多","做多","做多"],
})
# ── 條件表持久化：zlib 壓縮存 query_params + localStorage 雙重備援 ───────────
# 方案說明：
#   ① 主要：把條件表 JSON → zlib 壓縮 → base64 → query_params["tc"]
#      30 行條件表壓縮後約 460 字元，遠低於瀏覽器 URL 限制（~8000 字元）
#   ② 備援：同時透過 st.components.v1.html() 寫入瀏覽器 localStorage
#      換瀏覽器 / 清 URL 後仍可從 localStorage 恢復
#   恢復優先順序：localStorage（頁面載入 JS）→ query_params → 預設值

import zlib, base64

_TG_COLS = ["排名","異動標記","成交量標記","K線形態","回測勝率","方向"]
_LS_KEY  = "streamlit_tg_conds"   # localStorage key name


def _tg_encode(df: pd.DataFrame) -> str:
    """DataFrame → zlib壓縮 → base64 字串（URL 安全）。"""
    try:
        records = df[_TG_COLS].fillna("").to_dict(orient="records")
        raw     = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
        return base64.urlsafe_b64encode(
            zlib.compress(raw.encode("utf-8"), level=9)
        ).decode("ascii")
    except Exception:
        return ""


def _tg_decode(s: str) -> pd.DataFrame:
    """base64 字串 → zlib 解壓 → DataFrame。"""
    try:
        raw     = zlib.decompress(base64.urlsafe_b64decode(s)).decode("utf-8")
        records = json.loads(raw)
        df      = pd.DataFrame(records)
        for col in _TG_COLS:
            if col not in df.columns:
                df[col] = "做多" if col == "方向" else ""
        return df[_TG_COLS].fillna("")
    except Exception:
        return pd.DataFrame()


def _tg_save(df: pd.DataFrame):
    """
    同時把條件表寫入：
      ① query_params["tc"]（跨 session 恢復）
      ② 瀏覽器 localStorage（跨 URL 恢復）
    """
    encoded = _tg_encode(df)
    if not encoded:
        return
    # ① query_params
    try:
        st.query_params["tc"] = encoded
    except Exception:
        pass
    # ② localStorage（高度 0 的隱形 component）
    try:
        import streamlit.components.v1 as components
        _ls_js = f"""
        <script>
        try {{
            localStorage.setItem("{_LS_KEY}", {json.dumps(encoded)});
        }} catch(e) {{}}
        </script>
        """
        components.html(_ls_js, height=0, scrolling=False)
    except Exception:
        pass


def _tg_load_ls_component():
    """
    渲染一個隱形 component：
      讀取 localStorage 中的條件表，
      若存在且比 query_params 更新，則把值注入 query_params 並 rerun。
    只在 session 剛初始化（tg_conds 不在 session_state）時執行一次。
    """
    try:
        import streamlit.components.v1 as components
        _cur_qp = st.query_params.get("tc", "")
        _ls_read_js = f"""
        <script>
        (function() {{
            var ls = "";
            try {{ ls = localStorage.getItem("{_LS_KEY}") || ""; }} catch(e) {{}}
            var qp = {json.dumps(_cur_qp)};
            // 只有 localStorage 有值而 query_params 沒有時才注入
            if (ls && ls !== qp) {{
                var url = new URL(window.parent.location.href);
                url.searchParams.set("tc", ls);
                window.parent.history.replaceState(null, "", url.toString());
                // 觸發 Streamlit rerun（更新 URL 後 Streamlit 會自動 rerun）
                window.parent.location.reload();
            }}
        }})();
        </script>
        """
        components.html(_ls_read_js, height=0, scrolling=False)
    except Exception:
        pass


# ── 初始化順序：① query_params → ② 預設值 ───────────────────────────────────
if "tg_conds" not in st.session_state:
    # 若 query_params 沒有值，先讓 localStorage component 嘗試注入
    _qp_encoded = st.query_params.get("tc", "")
    if _qp_encoded:
        _restored = _tg_decode(_qp_encoded)
        if not _restored.empty:
            st.session_state["tg_conds"] = _restored
        else:
            st.session_state["tg_conds"] = _TG_DEFAULT.copy()
            _tg_load_ls_component()
    else:
        # 沒有 query_params → 嘗試從 localStorage 恢復
        st.session_state["tg_conds"] = _TG_DEFAULT.copy()
        _tg_load_ls_component()

# 向後相容：補齊缺少欄位
_cur = st.session_state["tg_conds"]
for _col in _TG_COLS:
    if _col not in _cur.columns:
        _cur[_col] = "做多" if _col == "方向" else ""
st.session_state["tg_conds"] = _cur

telegram_conditions = st.data_editor(
    st.session_state["tg_conds"],
    num_rows="dynamic",
    key="tg_editor",
    column_config={
        "排名":       st.column_config.TextColumn("排名", width="small"),
        "異動標記":   st.column_config.TextColumn("異動標記", width="large"),
        "成交量標記": st.column_config.SelectboxColumn("成交量標記",
                        options=["","放量","縮量","—"], width="small"),
        "K線形態":    st.column_config.TextColumn("K線形態", width="medium"),
        "回測勝率":   st.column_config.TextColumn("回測勝率", width="small",
                        help="由回測一鍵加入時自動填入"),
        "方向":       st.column_config.TextColumn("方向", width="small",
                        help="填入「做多」或「做空」；一鍵加入時自動帶入"),
    },
    use_container_width=True,
)
# FIX: data_editor 可能回傳 None 值（尤其 SelectboxColumn 空格），統一清理
_tc = telegram_conditions.copy()
for _col in _tc.columns:
    _tc[_col] = _tc[_col].where(_tc[_col].notna(), "").astype(str).str.strip()
st.session_state["tg_conds"] = _tc
# 持久化：把最新條件表寫入 query_params + localStorage
_tg_save(_tc)

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

            # ── Selected-signal push (user-chosen signals) ────────────────
            # 先建立「信號 → 方向」查找表（從條件表讀取）
            _live_tg_p1 = st.session_state.get("tg_conds", telegram_conditions)
            _sig_dir_map: dict = {}   # sig_str → "做多" | "做空" | ""
            for _, _p1_row in _live_tg_p1.iterrows():
                _p1_marks = str(_p1_row.get("異動標記", ""))
                _p1_dir   = str(_p1_row.get("方向", "")).strip()
                if _p1_dir not in ("做多", "做空"):
                    continue
                for _p1_sig in _p1_marks.split(","):
                    _s = _p1_sig.strip()
                    if _s and _s not in _sig_dir_map:
                        _sig_dir_map[_s] = _p1_dir

            for sig in selected_signals:
                if sig in K_list:
                    # 方向優先從條件表讀取，否則從 SELL_SIGNALS 推斷
                    _p1_dir_val = _sig_dir_map.get(sig, "")
                    if not _p1_dir_val:
                        _p1_dir_val = "做空" if sig in SELL_SIGNALS else "做多"
                    _p1_dir_label = "🔴 做空（賣出）" if _p1_dir_val == "做空" else "🟢 做多（買入）"

                    _msg = (
                        f"📡 信號提醒\n"
                        f"股票：{ticker} ({selected_interval})\n"
                        f"信號：{sig}\n"
                        f"操作方向：{_p1_dir_label}\n"
                        f"價格：${cur_price:.2f}\n"
                        f"RSI：{data['RSI'].iloc[-1]:.1f}  MACD：{data['MACD'].iloc[-1]:.3f}\n"
                        f"成交量：{_fmt_vol(data['Volume'].iloc[-1])}  "
                        f"({data['成交量標記'].iloc[-1]})"
                    )
                    if st.session_state.get("tg_enabled", True):
                        _ok, _err = send_telegram_alert(_msg)
                        if _ok:
                            st.toast(f"📡 Telegram 已推送：{sig} ({_p1_dir_label})", icon="✅")
                        else:
                            st.warning(f"⚠️ Telegram 推送失敗（{sig}）：{_err}")
                    else:
                        st.toast(f"🔕 {sig} 已匹配，Telegram 已關閉", icon="🔕")

            # ── Condition matching ─────────────────────────────────────────
            def _safe_str(v) -> str:
                if v is None:
                    return ""
                s = str(v).strip()
                return "" if s.lower() in ("none", "nan", "nat", "") else s

            # ── 比對模式切換按鈕 ───────────────────────────────────────────
            # 預設存在 session_state；每個 ticker 共用同一個 toggle
            if "tg_match_mode" not in st.session_state:
                st.session_state["tg_match_mode"] = "first"   # "first" or "all"

            _col_mode, _col_info = st.columns([1, 4])
            with _col_mode:
                _mode_label = (
                    "🔁 模式：**全部比對**（點擊切換為第一個）"
                    if st.session_state["tg_match_mode"] == "all"
                    else "1️⃣ 模式：**第一個匹配**（點擊切換為全部）"
                )
                if st.button(_mode_label, key=f"tg_mode_btn_{ticker}"):
                    st.session_state["tg_match_mode"] = (
                        "first" if st.session_state["tg_match_mode"] == "all" else "all"
                    )
                    st.rerun()
            with _col_info:
                if st.session_state["tg_match_mode"] == "all":
                    st.caption(
                        "🔁 **全部比對模式**：逐行掃描條件表，"
                        "每一個符合的條件都各自發一條 Telegram 訊息。"
                    )
                else:
                    st.caption(
                        "1️⃣ **第一個匹配模式**：從排名第 1 行開始比對，"
                        "找到第一個符合的條件就觸發並停止，不再往後比對。"
                    )

            _scan_all  = (st.session_state["tg_match_mode"] == "all")
            _live_tg   = st.session_state.get("tg_conds", telegram_conditions)
            _cur_vol   = _safe_str(data["成交量標記"].iloc[-1])
            _cur_kline = _safe_str(data["K線形態"].iloc[-1])

            # ── 預先計算各指標（發訊息時共用） ────────────────────────────
            _rsi_val  = float(data["RSI"].iloc[-1])
            _macd_val = float(data["MACD"].iloc[-1])
            _sig_line = float(data["Signal_Line"].iloc[-1]) if "Signal_Line" in data.columns else 0.0
            _vix_val  = data["VIX"].iloc[-1]
            _vix_str  = f"{_vix_val:.1f}" if pd.notna(_vix_val) else "N/A"
            _near_str = near_dense_info if near_dense else "無密集區靠近"
            _dir_icon = "🟢" if px_pct >= 0 else "🔴"
            _rsi_icon = "🔥" if _rsi_val > 70 else ("🧊" if _rsi_val < 30 else "⚪")
            _vol_icon = "📈" if _cur_vol == "放量" else "📉"

            def _build_tg_msg(rank: str, backtest_wr: str, match_no: int, total_matches: int,
                             direction: str = "做多") -> str:
                """組裝單條 Telegram 訊息，match_no/total_matches 用於全部比對模式的序號顯示。"""
                _header = (
                    f"{'='*28}\n"
                    f"🚨 Telegram 觸發條件匹配"
                    + (f"（第 {match_no}/{total_matches} 條）" if total_matches > 1 else "")
                    + f"\n{'='*28}"
                )
                _lines = [
                    _header, "",
                    f"股票代號  : {ticker}",
                    f"時間框架  : {selected_interval}",
                    f"觸發時間  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "--- 價格資訊 ---",
                    f"現價      : ${cur_price:.2f}  {_dir_icon} {px_pct:+.2f}%",
                    f"成交量    : {_fmt_vol(data['Volume'].iloc[-1])}  {_vol_icon} {_cur_vol}  ({v_pct:+.1f}%)",
                    "",
                    "--- 技術指標 ---",
                    f"RSI       : {_rsi_val:.1f}  {_rsi_icon}",
                    f"MACD      : {_macd_val:.4f}  (Signal: {_sig_line:.4f})",
                    f"K線形態   : {_cur_kline}",
                    f"VIX       : {_vix_str}",
                    f"密集區    : {_near_str}",
                    "",
                    "--- 觸發信號 ---",
                ]
                for _s in K_list[:12]:
                    _lines.append(f"  {_s}")
                if len(K_list) > 12:
                    _lines.append(f"  ... 共 {len(K_list)} 個信號")
                if direction == "做多":
                    _dir_label = "🟢 做多（買入）"
                elif direction == "做空":
                    _dir_label = "🔴 做空（賣出）"
                else:
                    _dir_label = "未設定（請在條件表填寫）"
                _lines += [
                    "",
                    "--- 匹配條件 ---",
                    f"條件排名  : #{rank}",
                    f"回測勝率  : {backtest_wr}",
                    f"方向      : {_dir_label}",
                    f"{'='*28}",
                ]
                return "\n".join(_lines)

            # ── 核心比對迴圈 ──────────────────────────────────────────────
            _matched_list = []   # list of (rank, wr, row_idx)

            for _ci, cond_row in _live_tg.iterrows():
                _raw_marks = _safe_str(cond_row.get("異動標記", ""))
                if not _raw_marks:
                    continue
                req = [s.strip() for s in _raw_marks.split(",") if s.strip()]
                if not req:
                    continue

                c_vol   = _safe_str(cond_row.get("成交量標記", ""))
                c_kline = _safe_str(cond_row.get("K線形態", ""))
                vol_ok    = (c_vol   == "") or (c_vol   in ("—", "全部")) or (c_vol   == _cur_vol)
                kline_ok  = (c_kline == "") or (c_kline in ("—", "全部")) or (c_kline == _cur_kline)
                signals_ok = all(s in K_list for s in req)

                if signals_ok and vol_ok and kline_ok:
                    _rank_raw = _safe_str(cond_row.get("排名", ""))
                    _rank     = _rank_raw if _rank_raw else f"#{_ci + 1}"
                    _wr_raw   = _safe_str(cond_row.get("回測勝率", ""))
                    _wr       = _wr_raw if _wr_raw else "N/A"
                    _dir_raw  = _safe_str(cond_row.get("方向", ""))
                    _dir      = _dir_raw if _dir_raw in ("做多", "做空") else ""
                    _matched_list.append((_rank, _wr, _ci, _dir))

                    if not _scan_all:
                        break   # 第一個匹配模式：找到就停

            # ── 發送所有匹配訊息 & UI 顯示 ───────────────────────────────
            _total_matched = len(_matched_list)

            if _total_matched == 0:
                pass   # 無匹配，靜默

            else:
                # UI 摘要欄
                if _total_matched == 1:
                    _rank0, _wr0, _, _dir0 = _matched_list[0]
                    _dir0_label = ("🟢 做多" if _dir0 == "做多"
                                   else "🔴 做空" if _dir0 == "做空"
                                   else "未設定")
                    st.info(
                        f"🎯 **條件匹配！** 排名 {_rank0}，回測勝率 {_wr0}，"
                        f"方向 {_dir0_label}\n\n"
                        f"信號：{K_str[:120]}\n"
                        f"成交量：{_cur_vol}  K線：{_cur_kline}",
                    )
                else:
                    _ranks_str = "、".join(r for r, _, __, ___ in _matched_list)
                    _dirs_str = "、".join(
                        ("🟢" if d=="做多" else "🔴" if d=="做空" else "—")
                        for _,_,__,d in _matched_list
                    )
                    st.info(
                        f"🎯 **共匹配 {_total_matched} 條條件！** 排名：{_ranks_str}\n\n"
                        f"方向：{_dirs_str}\n"
                        f"信號：{K_str[:120]}\n"
                        f"成交量：{_cur_vol}  K線：{_cur_kline}",
                    )

                # 逐一發送每條匹配訊息
                _send_ok_count  = 0
                _send_err_msgs  = []
                _tg_on = st.session_state.get("tg_enabled", True)

                if not _tg_on:
                    # 開關關閉：靜默，只顯示 UI 提示
                    _ranks_muted = "、".join(r for r,_,__,___ in _matched_list)
                    st.warning(
                        f"🔕 **Telegram 已關閉**，訊息未發送。\n"
                        f"匹配條件：排名 {_ranks_muted}（共 {_total_matched} 條）。\n"
                        f"完成調參後請點擊頂部「🔴 Telegram：已關閉」按鈕重新開啟。",
                        icon="🔕",
                    )
                else:
                    for _mn, (_rank, _wr, _ci, _dir) in enumerate(_matched_list, start=1):
                        _msg = _build_tg_msg(
                            rank=_rank, backtest_wr=_wr,
                            match_no=_mn, total_matches=_total_matched,
                            direction=_dir,
                        )
                        _ok, _err = send_telegram_alert(_msg)
                        if _ok:
                            _send_ok_count += 1
                            st.toast(
                                f"✅ {ticker} 條件 #{_rank} 匹配，Telegram 已推送"
                                + (f"（{_mn}/{_total_matched}）" if _total_matched > 1 else ""),
                                icon="📨",
                            )
                        else:
                            _send_err_msgs.append(f"排名 {_rank}：{_err}")

                    # 統一顯示發送結果
                    if _send_ok_count > 0 and not _send_err_msgs:
                        st.success(
                            f"📨 **Telegram 全部發送成功！** "
                            f"共 {_send_ok_count} 條訊息"
                            + (f"（排名 {', '.join(r for r,_,__,___ in _matched_list)}）"
                               if _total_matched > 1 else
                               f"（排名 {_matched_list[0][0]}，回測勝率 {_matched_list[0][1]}，"
                               f"方向 {'🟢 做多' if _matched_list[0][3]=='做多' else '🔴 做空' if _matched_list[0][3]=='做空' else '未設定'}）"),
                            icon="✅",
                        )
                    elif _send_ok_count > 0 and _send_err_msgs:
                        st.warning(
                            f"⚠️ 部分發送成功（{_send_ok_count}/{_total_matched}）。\n"
                            + "\n".join(_send_err_msgs),
                        )
                    else:
                        if not (BOT_TOKEN and CHAT_ID):
                            st.warning(
                                f"⚠️ 條件已匹配（{_total_matched} 條）但 **Telegram 未設定**，訊息未發送。\n\n"
                                f"請在 `.streamlit/secrets.toml` 中設定 `BOT_TOKEN` 和 `CHAT_ID`。",
                                icon="⚙️",
                            )
                        else:
                            st.error(
                                f"❌ **Telegram 全部發送失敗**（已匹配 {_total_matched} 條）：\n"
                                + "\n".join(_send_err_msgs),
                                icon="🚨",
                            )


            # ── Breakout / Breakdown alerts ────────────────────────────────
            _tg_on_bo = st.session_state.get("tg_enabled", True)

            if pd.notna(data["High_Max"].iloc[-1]) and data["High"].iloc[-1] >= data["High_Max"].iloc[-1]:
                _bo_msg = (
                    f"🚀 突破新高提醒\n"
                    f"股票：{ticker} ({selected_interval})\n"
                    f"現價 ${data['High'].iloc[-1]:.2f} 創 {W} 根K線新高\n"
                    f"成交量：{_fmt_vol(data['Volume'].iloc[-1])}  ({_cur_vol})\n"
                    f"方向：🟢 做多（買入）"
                )
                if _tg_on_bo:
                    _ok, _err = send_telegram_alert(_bo_msg)
                    if _ok:
                        st.toast(f"🚀 {ticker} 破 {W}K 新高，Telegram 已推送", icon="🚀")
                    else:
                        st.warning(f"⚠️ {ticker} 突破新高 Telegram 推送失敗：{_err}")
                else:
                    st.toast(f"🔕 {ticker} 破 {W}K 新高（Telegram 已關閉）", icon="🔕")

            if pd.notna(data["Low_Min"].iloc[-1]) and data["Low"].iloc[-1] <= data["Low_Min"].iloc[-1]:
                _bd_msg = (
                    f"🔻 跌破新低提醒\n"
                    f"股票：{ticker} ({selected_interval})\n"
                    f"現價 ${data['Low'].iloc[-1]:.2f} 創 {W} 根K線新低\n"
                    f"成交量：{_fmt_vol(data['Volume'].iloc[-1])}  ({_cur_vol})\n"
                    f"方向：🔴 做空（賣出）"
                )
                if _tg_on_bo:
                    _ok, _err = send_telegram_alert(_bd_msg)
                    if _ok:
                        st.toast(f"🔻 {ticker} 穿 {W}K 新低，Telegram 已推送", icon="🔻")
                    else:
                        st.warning(f"⚠️ {ticker} 跌破新低 Telegram 推送失敗：{_err}")
                else:
                    st.toast(f"🔕 {ticker} 穿 {W}K 新低（Telegram 已關閉）", icon="🔕")


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
    # ══════════════════════════════════════════════════════════════════════════
    #  BACKTEST TAB  v2: 3 independent dimensions
    # ══════════════════════════════════════════════════════════════════════════
    st.header("🔬 回測：三維信號勝率分析")

    st.info(
        "**三個維度分開計算，找出歷史勝率最高的組合**\n\n"
        "| 維度 | 說明 |\n"
        "|------|------|\n"
        "| 📊 信號組合 | 多個技術指標同時出現（基礎維度）|\n"
        "| 📦 信號+成交量 | 信號組合 × 放量/縮量 |\n"
        "| 🕯️ 信號+K線形態 | 信號組合 × K線形態（大陽線、錘子線…）|\n\n"
        "三個維度**各自獨立計算**，讓您分別看到量能與K線結構如何提升勝率。\n"
        "⚠️ 回測僅供參考，請結合風險管理進行決策。"
    )

    # ── Parameters ────────────────────────────────────────────────────────────
    bt_ticker = st.selectbox("選擇回測股票", selected_tickers, key="bt_ticker")

    # ── yfinance 短週期限制說明 ────────────────────────────────────────────
    # 1m  : 最近 7 天   → 約  2,730 根（含盤前後）/ 交易時段約 390 根/天
    # 5m  : 最近 60 天  → 約  2,340 根
    # 15m : 最近 60 天  → 約    780 根
    # 30m : 最近 60 天  → 約    390 根
    # 1h  : 最近 730天  → 約  1,430 根
    # 1d  : 最多 max    → 約  5,000 根（視股票上市年數）
    # 1wk : 最多 max    → 約  1,300 根
    # 1mo : 最多 max    → 約    300 根

    # 每個間隔允許的 period 清單（受 yfinance 限制）
    _INTERVAL_PERIOD_MAP = {
        "1m":  {"periods": ["1d","5d","7d"],
                "default": "7d",
                "help": "1m 間隔最多只能拉取最近 7 天，約 2,730 根K線"},
        "5m":  {"periods": ["5d","1mo","60d"],
                "default": "60d",
                "help": "5m 間隔最多只能拉取最近 60 天，約 2,340 根K線"},
        "15m": {"periods": ["5d","1mo","60d"],
                "default": "60d",
                "help": "15m 間隔最多只能拉取最近 60 天，約 780 根K線"},
        "30m": {"periods": ["5d","1mo","60d"],
                "default": "60d",
                "help": "30m 間隔最多只能拉取最近 60 天，約 390 根K線"},
        "1h":  {"periods": ["1mo","3mo","6mo","1y","2y"],
                "default": "2y",
                "help": "1h 間隔最多只能拉取最近 730 天，約 1,430 根K線"},
        "1d":  {"periods": ["3mo","6mo","1y","2y","5y","ytd","max"],
                "default": "2y",
                "help": "日線最多可拉取全部歷史，建議 2y 以上"},
        "1wk": {"periods": ["6mo","1y","2y","5y","ytd","max"],
                "default": "5y",
                "help": "週線樣本較少，建議 5y 以上"},
        "1mo": {"periods": ["1y","2y","5y","ytd","max"],
                "default": "max",
                "help": "月線樣本極少，建議 max"},
    }

    # 所有可選間隔
    _all_intervals = ["1m","5m","15m","30m","1h","1d","1wk","1mo"]

    st.caption(
        f"ℹ️ 主監控目前：**{selected_period}** / **{selected_interval}**　"
        "回測可獨立選擇時間範圍與K線間隔。"
    )

    _col_i, _col_p = st.columns(2)

    # 先選間隔，再動態更新可用的 period 清單
    bt_interval = _col_i.selectbox(
        "回測K線間隔",
        _all_intervals,
        index=_all_intervals.index("1d"),
        key="bt_interval",
        help="短週期間隔（1m/5m/15m/30m/1h）受 yfinance 限制，可回測天數較少",
    )

    _period_cfg  = _INTERVAL_PERIOD_MAP.get(bt_interval, _INTERVAL_PERIOD_MAP["1d"])
    _period_opts = _period_cfg["periods"]
    _period_def  = _period_cfg["default"]
    _period_idx  = _period_opts.index(_period_def) if _period_def in _period_opts else 0

    bt_period = _col_p.selectbox(
        "回測時間範圍",
        _period_opts,
        index=_period_idx,
        key="bt_period",
        help=_period_cfg["help"],
    )

    # ── 預估K線數量（含短週期） ────────────────────────────────────────────
    # 每交易日：1m=390根, 5m=78根, 15m=26根, 30m=13根, 1h=6.5根
    _BARS_PER_DAY = {"1m":390, "5m":78, "15m":26, "30m":13,
                     "1h":7,   "1d":1,  "1wk":1,  "1mo":1}
    _PERIOD_DAYS  = {
        "1d":1, "5d":5, "7d":7, "60d":60, "1mo":21, "3mo":63,
        "6mo":126, "1y":252, "2y":504, "5y":1260, "ytd":180, "max":5000,
    }
    _days = _PERIOD_DAYS.get(bt_period, 0)
    _bpd  = _BARS_PER_DAY.get(bt_interval, 1)
    _est_n = _days * _bpd if _days and _bpd else "?"

    _est_icon = ("🟢" if isinstance(_est_n, int) and _est_n >= 100
                 else "🟡" if isinstance(_est_n, int) and _est_n >= 30
                 else "🔴")
    _est_warn = ""
    if bt_interval in ("1m","5m","15m","30m"):
        _est_warn = "　⚠️ 短週期樣本有限，勝率統計建議搭配較低的「最少出現次數」"
    elif isinstance(_est_n, int) and _est_n < 100:
        _est_warn = "　（建議 ≥ 100 根，樣本越多勝率越可信）"

    st.caption(
        f"{_est_icon} 預估約 **{_est_n}** 根K線{_est_warn}"
    )

    # ── 短週期額外說明 ─────────────────────────────────────────────────────
    if bt_interval in ("1m","5m","15m","30m","1h"):
        _help_txt = _period_cfg["help"]
        st.info(
            f"📌 {bt_interval} 短週期回測說明\n\n"
            f"- yfinance 限制：{_help_txt}\n"
            "- 短週期信號觸發更頻繁，建議將最少出現次數降至 3～5\n"
            "- 短週期雜訊較多，建議搭配信號+成交量或信號+K線形態維度篩選\n"
            "- 勝率數據僅反映歷史規律，短週期市場結構變化快，請謹慎使用"
        )

    col_a, col_b, col_c = st.columns(3)
    bt_min    = col_a.number_input("最少信號組合數", 2, 3, int(BT_MIN_COMBO), 1, key="bt_min")
    bt_max    = col_b.number_input("最多信號組合數", 2, 5, int(BT_MAX_COMBO), 1, key="bt_max")
    bt_occ    = col_c.number_input("最少出現次數",   2, 20, int(BT_MIN_OCC),  1, key="bt_occ")

    col_d, col_e, _ = st.columns([1, 1, 1])
    bt_wr_thr  = col_d.number_input(
        "高勝率閾值 (%)", 50, 95, 60, 5, key="bt_wr_thr",
        help="高於此值才列入高勝率區，並可一鍵加入 Telegram 條件",
    )
    bt_pnl_thr = col_e.number_input(
        "最低平均盈虧 (%)", -10.0, 20.0, 0.0, 0.1,
        key="bt_pnl_thr",
        format="%.1f",
        help="同時滿足勝率閾值 且 平均盈虧 ≥ 此值才列入高勝率區。設為 0 = 不限制負盈虧",
    )

    if st.button("🚀 開始回測", type="primary"):
        with st.spinner(f"正在計算 {bt_ticker}（{bt_period} / {bt_interval}）三維勝率，稍候…"):
            try:
                # ── Prepare data（使用回測專屬時間範圍，與主監控無關）──────
                bt_data = yf.Ticker(bt_ticker).history(
                    period=bt_period, interval=bt_interval).reset_index()
                if "Date" in bt_data.columns:
                    bt_data = bt_data.rename(columns={"Date": "Datetime"})
                bt_data["Datetime"] = pd.to_datetime(bt_data["Datetime"]).dt.tz_localize(None)
                if len(bt_data) < 30:
                    n_bars = len(bt_data)
                    st.warning(f"資料不足（{n_bars} 根K線 < 30）。目前：{bt_period} / {bt_interval}，請選更長時間範圍。")
                    st.stop()

                bt_data["前5均量"]         = bt_data["Volume"].rolling(5).mean()
                bt_data["Price Change %"]  = bt_data["Close"].pct_change() * 100
                bt_data["Volume Change %"] = bt_data["Volume"].pct_change() * 100
                bt_data["MACD"], bt_data["Signal_Line"], _ = calculate_macd(bt_data)
                bt_data["RSI"] = calculate_rsi(bt_data)
                for span, name in [(5,"EMA5"),(10,"EMA10"),(30,"EMA30"),(40,"EMA40")]:
                    bt_data[name] = bt_data["Close"].ewm(span=span, adjust=False).mean()
                bt_data["SMA50"]  = bt_data["Close"].rolling(50).mean()
                bt_data["SMA200"] = bt_data["Close"].rolling(200).mean()
                bt_data["VWAP"]   = calculate_vwap(bt_data)
                bt_data["MFI"]    = calculate_mfi(bt_data)
                bt_data["OBV"]    = calculate_obv(bt_data)
                bt_data["Up"]   = (bt_data["Close"] > bt_data["Close"].shift(1)).astype(int)
                bt_data["Down"] = (bt_data["Close"] < bt_data["Close"].shift(1)).astype(int)
                bt_data["Continuous_Up"]   = bt_data["Up"] * (
                    bt_data["Up"].groupby((bt_data["Up"] == 0).cumsum()).cumcount() + 1)
                bt_data["Continuous_Down"] = bt_data["Down"] * (
                    bt_data["Down"].groupby((bt_data["Down"] == 0).cumsum()).cumcount() + 1)
                W2 = int(MFI_WIN)
                bt_data["High_Max"]       = bt_data["High"].rolling(W2).max()
                bt_data["Low_Min"]        = bt_data["Low"].rolling(W2).min()
                bt_data["Close_Roll_Max"] = bt_data["Close"].rolling(W2).max()
                bt_data["Close_Roll_Min"] = bt_data["Close"].rolling(W2).min()
                bt_data["MFI_Roll_Max"]   = bt_data["MFI"].rolling(W2).max()
                bt_data["MFI_Roll_Min"]   = bt_data["MFI"].rolling(W2).min()
                bt_data["MFI_Bear_Div"]   = (
                    (bt_data["Close"] == bt_data["Close_Roll_Max"]) &
                    (bt_data["MFI"] < bt_data["MFI_Roll_Max"].shift(1)))
                bt_data["MFI_Bull_Div"]   = (
                    (bt_data["Close"] == bt_data["Close_Roll_Min"]) &
                    (bt_data["MFI"] > bt_data["MFI_Roll_Min"].shift(1)))
                bt_data["OBV_Roll_Max"] = bt_data["OBV"].rolling(20).max()
                bt_data["OBV_Roll_Min"] = bt_data["OBV"].rolling(20).min()
                for _nc in ["VIX","VIX_EMA_Fast","VIX_EMA_Slow",
                             "📈 股價漲跌幅(%)","📊 成交量變動幅(%)",
                             "Close_N_High","Close_N_Low"]:
                    bt_data[_nc] = np.nan

                data = bt_data  # closure for compute_all_signals
                bt_data["異動標記"] = compute_all_signals(bt_data, PARAMS)

                # K線形態 & 成交量標記
                _buster2 = str(round(float(bt_data["Close"].iloc[-1]), 4))
                kdf2 = get_kline_patterns(bt_ticker, bt_period, bt_interval,
                                          BODY_RATIO_TH, SHADOW_RATIO_TH, DOJI_BODY_TH, _buster2)
                kdf2["Datetime"] = pd.to_datetime(kdf2["Datetime"]).dt.tz_localize(None)
                bt_data = bt_data.merge(kdf2, on="Datetime", how="left")
                bt_data["K線形態"]  = bt_data["K線形態"].fillna("普通K線")
                bt_data["成交量標記"] = bt_data.apply(
                    lambda r: "放量" if pd.notna(r["前5均量"]) and r["Volume"] > r["前5均量"]
                    else "縮量", axis=1)

                # Save full enriched data for detail validation
                st.session_state["bt_raw_data"] = bt_data.copy()

                _kw = dict(min_combo=int(bt_min), max_combo=int(bt_max), min_occ=int(bt_occ))

                # ── Run 3 independent dimensions ──────────────────────────
                df_sig  = _base_signal_combos(bt_data, **_kw)
                df_vol  = _signal_x_volume_combos(bt_data, **_kw)
                df_kl   = _signal_x_kline_combos(bt_data, **_kw)

                st.session_state["bt_df_sig"]      = df_sig
                st.session_state["bt_df_vol"]      = df_vol
                st.session_state["bt_df_kl"]       = df_kl
                st.session_state["_result_wr_thr"]      = int(bt_wr_thr)
                st.session_state["_result_pnl_thr"]    = float(bt_pnl_thr)
                st.session_state["_result_ticker"]  = bt_ticker
                st.session_state["_result_period"]  = bt_period
                st.session_state["_result_interval"]= bt_interval
                st.session_state["_result_total_bars"]  = len(bt_data)

            except Exception as e:
                st.error(f"回測失敗：{e}")
                with st.expander("詳細錯誤"):
                    st.code(traceback.format_exc())

    # ── Results (persist via session_state) ───────────────────────────────────
    if "bt_df_sig" in st.session_state:
        df_sig  = st.session_state["bt_df_sig"]
        df_vol  = st.session_state["bt_df_vol"]
        df_kl   = st.session_state["bt_df_kl"]
        _wr_thr          = st.session_state.get("_result_wr_thr", 60)
        _pnl_thr         = st.session_state.get("_result_pnl_thr", 0.0)
        _bt_lbl          = st.session_state.get("_result_ticker",   bt_ticker)
        _bt_period_used  = st.session_state.get("_result_period",   "?")
        _bt_interval_used= st.session_state.get("_result_interval", "?")
        _bt_bars_used    = st.session_state.get("_result_total_bars",   "?")

        # Show what data was actually used for this backtest run
        st.info(
            f"📊 本次回測使用資料：**{_bt_lbl}**　"
            f"時間範圍：**{_bt_period_used}**　"
            f"K線間隔：**{_bt_interval_used}**　"
            f"共 **{_bt_bars_used}** 根K線"
            + ("　⚠️ 樣本偏少，勝率僅供參考" if isinstance(_bt_bars_used, int) and _bt_bars_used < 100 else "")
        )

        # ── Helper: render one dimension result ───────────────────────────
        def _render_dim(df_dim: pd.DataFrame, title: str, wr_thr: int,
                        col_order: list, dim_key: str, pnl_thr: float = 0.0):
            if df_dim.empty:
                st.warning(f"{title}：無有效組合，請增加時間範圍或降低最少出現次數。")
                return

            # 篩選高勝率：勝率 ≥ wr_thr 且 平均盈虧 ≥ pnl_thr
            hi = df_dim[df_dim["勝率(%)"] >= wr_thr].copy()
            if "平均盈虧(%)" in hi.columns and pnl_thr != 0.0:
                hi = hi[hi["平均盈虧(%)"] >= pnl_thr]

            total = len(df_dim)
            _pnl_cond_str = f" 且 平均盈虧 ≥ {pnl_thr}%" if pnl_thr != 0.0 else ""
            st.success(
                f"✅ {title}：找到 **{total}** 組，"
                f"其中 **{len(hi)}** 組勝率 ≥ {wr_thr}%{_pnl_cond_str}"
            )

            # Summary row
            m1, m2, m3 = st.columns(3)
            m1.metric("最高勝率",   f"{df_dim['勝率(%)'].max():.1f}%")
            m2.metric("平均勝率",   f"{df_dim['勝率(%)'].mean():.1f}%")
            m3.metric(f"≥{wr_thr}%", len(hi))

            # High win-rate table
            if not hi.empty:
                disp_cols = [c for c in col_order if c in hi.columns]
                st.dataframe(
                    hi[disp_cols].style.background_gradient(subset=["勝率(%)"], cmap="Greens", gmap=hi["勝率(%)"].apply(pd.to_numeric, errors="coerce")),
                    use_container_width=True,
                    height=min(400, 38 * (len(hi) + 1) + 40),
                )

                # ── ONE-CLICK ADD button ───────────────────────────────────
                btn_label = f"➕ 一鍵加入 {title} 高勝率組合到 Telegram 條件"
                if st.button(btn_label, key=f"add_{dim_key}", type="primary"):
                    _one_click_add(hi, dim_key)

            # ── Detail validation & CSV export ────────────────────────────
            with st.expander(f"🔬 {title} 詳細驗證 & CSV 下載（點擊展開）"):
                st.caption(
                    "選擇一個組合，系統逐筆列出每次信號出現後的完整交易記錄，"
                    "包含進出場價、盈虧%、最大順逆勢，並計算統計摘要驗證勝率準確度。"
                )
                # Build combo choices from high-wr rows (or all if hi is empty)
                _source_df = hi if not hi.empty else df_dim
                _combo_choices = []
                for _, _r in _source_df.iterrows():
                    _lbl = _r["信號組合"]
                    if _r.get("成交量標記","—") != "—":
                        _lbl += f"  [{_r['成交量標記']}]"
                    if _r.get("K線形態","—") != "—":
                        _lbl += f"  [{_r['K線形態']}]"
                    _lbl += f"  ({_r['勝率(%)']}%  {_r['出現次數']}次)"
                    _combo_choices.append(_lbl)

                if not _combo_choices:
                    st.info("無可選組合，請先完成回測。")
                else:
                    _sel = st.selectbox("選擇要驗證的組合", _combo_choices,
                                        key=f"detail_sel_{dim_key}")
                    _sel_idx = _combo_choices.index(_sel)
                    _sel_row = _source_df.iloc[_sel_idx]

                    _hold_bars = st.number_input("持倉根數（幾根K線後出場）",
                                                  min_value=1, max_value=20, value=1, step=1,
                                                  key=f"hold_{dim_key}",
                                                  help="1 = 信號出現後的下一根K線收盤出場")

                    if st.button(f"📊 展開逐筆交易記錄", key=f"detail_btn_{dim_key}"):
                        _bt_raw = st.session_state.get("bt_raw_data")
                        if _bt_raw is None:
                            st.warning("請重新點擊「🚀 開始回測」以載入原始資料。")
                        else:
                            with st.spinner("計算中..."):
                                _detail = _detailed_backtest(
                                    _bt_raw,
                                    signal_combo  = _sel_row["信號組合"],
                                    vol_filter    = _sel_row.get("成交量標記","—"),
                                    kline_filter  = _sel_row.get("K線形態","—"),
                                    direction     = _sel_row.get("方向","做多"),
                                    hold_bars     = int(_hold_bars),
                                )

                            if _detail.empty:
                                st.warning("此組合在所選資料中無完整交易記錄（可能樣本不足或出場K線超出範圍）。")
                            else:
                                _stats = _summary_stats(_detail)

                                # ── Stats cards ──────────────────────────
                                st.subheader("📈 統計摘要（驗證勝率準確度）")
                                _sc = st.columns(4)
                                _sc[0].metric("實際勝率",
                                              f"{_stats.get('實際勝率(%)','N/A')}%",
                                              help="與回測勝率一致即代表計算正確")
                                _sc[1].metric("期望值/筆",
                                              f"{_stats.get('期望值每筆(%)','N/A')}%",
                                              help=">0 代表長期有正期望值")
                                _sc[2].metric("獲利因子",
                                              str(_stats.get("獲利因子","N/A")),
                                              help=">1.5 為優質策略")
                                _sc[3].metric("最大回撤",
                                              f"{_stats.get('最大回撤(%)','N/A')}%",
                                              help="累計盈虧序列的最大跌幅")

                                _sc2 = st.columns(4)
                                _sc2[0].metric("總筆數",      _stats.get("總交易筆數","N/A"))
                                _sc2[1].metric("平均盈利/筆", f"{_stats.get('平均盈利(%)','N/A')}%")
                                _sc2[2].metric("平均虧損/筆", f"{_stats.get('平均虧損(%)','N/A')}%")
                                _sc2[3].metric("盈虧比",       str(_stats.get("盈虧比","N/A")))

                                _sc3 = st.columns(3)
                                _sc3[0].metric("累計盈虧",
                                               f"{_stats.get('累計盈虧(%)','N/A')}%")
                                _sc3[1].metric("最大連勝",
                                               _stats.get("最大連勝","N/A"))
                                _sc3[2].metric("最大連敗",
                                               _stats.get("最大連敗","N/A"))

                                # ── Cumulative PnL chart ──────────────────
                                _fig_pnl = go.Figure()
                                _fig_pnl.add_trace(go.Scatter(
                                    x=_detail["序號"],
                                    y=_detail["累計盈虧(%)"],
                                    mode="lines+markers",
                                    name="累計盈虧",
                                    line=dict(color="#2ecc71", width=2),
                                    fill="tozeroy",
                                    fillcolor="rgba(46,204,113,0.12)",
                                ))
                                _fig_pnl.add_hline(y=0, line_dash="dash",
                                                   line_color="gray", line_width=0.8)
                                _fig_pnl.update_layout(
                                    title="累計盈虧曲線（等額投入模擬）",
                                    xaxis_title="交易序號",
                                    yaxis_title="累計盈虧 (%)",
                                    template="plotly_dark", height=320,
                                    margin=dict(l=40,r=40,t=50,b=40),
                                )
                                st.plotly_chart(_fig_pnl, use_container_width=True,
                                                key=f"pnl_{dim_key}")

                                # ── Per-trade bar chart ───────────────────
                                _fig_bar = go.Figure(go.Bar(
                                    x=_detail["序號"],
                                    y=_detail["盈虧(%)"],
                                    marker_color=[
                                        "#2ecc71" if v >= 0 else "#e74c3c"
                                        for v in _detail["盈虧(%)"]
                                    ],
                                    name="單筆盈虧",
                                ))
                                _fig_bar.add_hline(y=0, line_dash="dash",
                                                   line_color="gray", line_width=0.8)
                                _fig_bar.update_layout(
                                    title="逐筆盈虧分佈",
                                    xaxis_title="交易序號",
                                    yaxis_title="盈虧 (%)",
                                    template="plotly_dark", height=280,
                                    margin=dict(l=40,r=40,t=50,b=40),
                                )
                                st.plotly_chart(_fig_bar, use_container_width=True,
                                                key=f"bar_{dim_key}")

                                # ── Detail table ──────────────────────────
                                st.subheader("📋 逐筆交易記錄")
                                _disp_detail = _detail[[
                                    "序號","進場時間","出場時間","持倉根數",
                                    "進場價","出場價","方向",
                                    "盈虧(%)","勝負",
                                    "最大順勢(%)","最大逆勢(%)",
                                    "累計盈虧(%)",
                                    "進場RSI","進場MACD",
                                    "成交量標記","K線形態",
                                    "連勝數","連敗數","觸發信號",
                                ]]
                                _color_map = {
                                    True:  "background-color: rgba(46,204,113,0.15)",
                                    False: "background-color: rgba(231,76,60,0.15)",
                                }
                                def _row_color(row):
                                    is_win = row["勝負"] == "✅ 勝"
                                    return [_color_map[is_win]] * len(row)
                                try:
                                    _styled = _disp_detail.style.apply(_row_color, axis=1)
                                    st.dataframe(_styled, use_container_width=True,
                                                 height=min(600, 35*(len(_disp_detail)+1)+40))
                                except Exception:
                                    st.dataframe(_disp_detail, use_container_width=True)

                                # ── CSV export ────────────────────────────
                                _combo_safe = (_sel_row["信號組合"][:40]
                                               .replace(" + ","_")
                                               .replace("/","_")
                                               .replace(" ","_"))
                                _ticker_safe = st.session_state.get("_result_ticker","stock")
                                _fname = (f"{_ticker_safe}_{_combo_safe}"
                                          f"_hold{int(_hold_bars)}"
                                          f"_{datetime.now().strftime('%Y%m%d')}.csv")

                                # Build full CSV with stats header
                                _stats_rows = [
                                    ["=== 統計摘要 ==="],
                                    ["股票", _ticker_safe],
                                    ["信號組合", _sel_row["信號組合"]],
                                    ["成交量篩選", _sel_row.get("成交量標記","—")],
                                    ["K線形態篩選", _sel_row.get("K線形態","—")],
                                    ["方向", _sel_row.get("方向","做多")],
                                    ["持倉根數", int(_hold_bars)],
                                    ["回測時間範圍", st.session_state.get("_result_period","?")],
                                    ["K線間隔", st.session_state.get("_result_interval","?")],
                                    ["總K線根數", st.session_state.get("_result_total_bars","?")],
                                    [],
                                ]
                                for k, v in _stats.items():
                                    _stats_rows.append([k, v])
                                _stats_rows.append([])
                                _stats_rows.append(["=== 逐筆交易記錄 ==="])

                                import io, csv as _csv
                                _buf = io.StringIO()
                                _w   = _csv.writer(_buf)
                                for _srow in _stats_rows:
                                    _w.writerow(_srow)
                                _detail.to_csv(_buf, index=False)
                                _csv_bytes = _buf.getvalue().encode("utf-8-sig")

                                st.download_button(
                                    label="📥 下載完整逐筆交易 CSV",
                                    data=_csv_bytes,
                                    file_name=_fname,
                                    mime="text/csv",
                                    type="primary",
                                )
                                st.caption(
                                    f"CSV 包含：統計摘要（{len(_stats)} 項指標）"
                                    f" + 逐筆記錄（{len(_detail)} 筆）"
                                )

            # Full table (collapsed)
            with st.expander(f"📊 {title} 全部 {total} 組（展開查看）"):
                disp_all = [c for c in col_order if c in df_dim.columns]
                st.dataframe(
                    df_dim[disp_all].style.background_gradient(subset=["勝率(%)"], cmap="RdYlGn", gmap=df_dim["勝率(%)"].apply(pd.to_numeric, errors="coerce")),
                    use_container_width=True, height=420,
                )

            # Bar chart: top 12
            top12 = df_dim.head(12).copy()
            y_labels = []
            for _, r in top12.iterrows():
                vol_part   = f" [{r['成交量標記']}]" if r.get("成交量標記","—") != "—" else ""
                kline_part = f" [{r['K線形態']}]"   if r.get("K線形態","—")   != "—" else ""
                y_labels.append(r["信號組合"] + vol_part + kline_part)
            bar_colors = ["#2ecc71" if d=="做多" else "#e74c3c" for d in top12["方向"]]
            fig_d = go.Figure(go.Bar(
                x=top12["勝率(%)"], y=y_labels, orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.1f}% ({n}次)" for v,n in zip(top12["勝率(%)"],top12["出現次數"])],
                textposition="outside",
            ))
            fig_d.add_vline(x=_wr_thr, line_dash="dash", line_color="gold",
                            annotation_text=f"{_wr_thr}%")
            fig_d.update_layout(
                title=f"{_bt_lbl} — {title}（前12）",
                xaxis_title="勝率 (%)", xaxis_range=[0, 115],
                height=520, template="plotly_dark",
                margin=dict(l=380, r=60, t=50, b=30),
            )
            st.plotly_chart(fig_d, use_container_width=True, key=f"chart_{dim_key}")

        # ── One-click add helper (dedup + re-rank) ────────────────────────
        def _one_click_add(hi_df: pd.DataFrame, dim_key: str):
            """Append high-WR rows to tg_conds, dedup, re-rank by 勝率."""
            existing = st.session_state.get("tg_conds", pd.DataFrame()).copy()
            if "回測勝率" not in existing.columns:
                existing["回測勝率"] = "N/A"

            new_rows = []
            for _, row in hi_df.iterrows():
                vol   = row.get("成交量標記","—")
                kl    = row.get("K線形態","普通K線")
                new_rows.append({
                    "排名":       "",
                    "異動標記":   row["信號組合"].replace(" + ", ", "),
                    "成交量標記": "—" if vol == "—" else vol,
                    "K線形態":    kl,
                    "回測勝率":   f"{row['勝率(%)']:.1f}%",
                    "方向":       row.get("方向", "做多"),
                })
            new_df = pd.DataFrame(new_rows)

            combined = pd.concat([existing, new_df], ignore_index=True)
            # Dedup: same 異動標記+成交量標記+K線形態 → keep last (new wins)
            combined = combined.drop_duplicates(
                subset=["異動標記","成交量標記","K線形態"], keep="last")

            # Re-rank by 勝率 descending
            def _parse(v):
                try:
                    return float(str(v).replace("%","").strip())
                except Exception:
                    return 0.0
            combined["_n"] = combined["回測勝率"].apply(_parse)
            combined = combined.sort_values("_n", ascending=False).drop(columns=["_n"])
            combined = combined.reset_index(drop=True)
            combined["排名"] = [str(i+1) for i in range(len(combined))]

            if "方向" not in combined.columns:
                combined["方向"] = ""
            _oc_saved = combined[
                ["排名","異動標記","成交量標記","K線形態","回測勝率","方向"]]
            st.session_state["tg_conds"] = _oc_saved
            _tg_save(_oc_saved)

            added = len(combined) - len(
                existing.drop_duplicates(subset=["異動標記","成交量標記","K線形態"]))
            _added_n = max(added, 0)
            st.success(
                f"✅ 已追加 **{_added_n}** 條新組合（去重後共 **{len(combined)}** 條）。\n\n"
                "📋 請捲動至頁面頂部的「**Telegram 觸發條件配置**」表格查看。\n\n"
                "系統每次刷新時，會自動比對最新一根K線是否符合條件表中的任何一條。\n"
                "一旦匹配，立即透過 Telegram 發送包含「現價、信號、RSI、MACD、"
                "K線形態、回測勝率」的完整交易信號。"
            )
            if _added_n == 0:
                st.info("ℹ️ 所有高勝率組合均已存在條件表中（無新增），去重後保留最新版本。")

        # ── Render 3 dimensions in tabs ────────────────────────────────────
        dim_tab1, dim_tab2, dim_tab3 = st.tabs([
            "📊 維度一：信號組合",
            "📦 維度二：信號 + 成交量",
            "🕯️ 維度三：信號 + K線形態",
        ])

        COLS_SIG  = ["信號組合","信號數量","勝率(%)","平均盈虧(%)","出現次數","方向"]
        COLS_VOL  = ["信號組合","成交量標記","信號數量","勝率(%)","平均盈虧(%)","出現次數","方向"]
        COLS_KL   = ["信號組合","K線形態","信號數量","勝率(%)","平均盈虧(%)","出現次數","方向"]

        with dim_tab1:
            _render_dim(df_sig,  f"{_bt_lbl} 信號組合",    _wr_thr, COLS_SIG, "sig", pnl_thr=_pnl_thr)
        with dim_tab2:
            _render_dim(df_vol,  f"{_bt_lbl} 信號+成交量", _wr_thr, COLS_VOL, "vol", pnl_thr=_pnl_thr)
        with dim_tab3:
            _render_dim(df_kl,   f"{_bt_lbl} 信號+K線形態",_wr_thr, COLS_KL,  "kl",  pnl_thr=_pnl_thr)

        # ── Best combo summary across all 3 dims ──────────────────────────
        st.markdown("---")
        st.subheader("💡 三維綜合最佳建議")
        all_hi = []
        for df_d, lbl in [(df_sig,"信號組合"),(df_vol,"信號+成交量"),(df_kl,"信號+K線形態")]:
            hi_d = df_d[df_d["勝率(%)"] >= _wr_thr] if not df_d.empty else pd.DataFrame()
            if not hi_d.empty and "平均盈虧(%)" in hi_d.columns and _pnl_thr != 0.0:
                hi_d = hi_d[hi_d["平均盈虧(%)"] >= _pnl_thr]
            if not hi_d.empty:
                best_row = hi_d.iloc[0].copy()
                best_row["_dim"] = lbl
                all_hi.append(best_row)

        if all_hi:
            overall_best = max(all_hi, key=lambda r: r["勝率(%)"])
            vol_info   = (f"  成交量：**{overall_best.get('成交量標記','—')}**"
                          if overall_best.get("成交量標記","—") != "—" else "")
            kline_info = (f"  K線形態：**{overall_best.get('K線形態','—')}**"
                          if overall_best.get("K線形態","—") != "—" else "")
            _best_pnl  = overall_best.get("平均盈虧(%)", "N/A")
            _pnl_icon  = ("📈" if isinstance(_best_pnl, (int, float)) and _best_pnl > 0
                          else "📉" if isinstance(_best_pnl, (int, float)) and _best_pnl < 0
                          else "—")
            st.success(
                f"🏆 **全局最佳組合**（來自「{overall_best['_dim']}」維度）\n\n"
                f"📊 信號：**{overall_best['信號組合']}**\n\n"
                f"{vol_info}{kline_info}\n\n"
                f"- 歷史勝率：**{overall_best['勝率(%)']}%**"
                f"  |  平均盈虧：**{_pnl_icon} {_best_pnl}%**"
                f"  |  出現次數：**{overall_best['出現次數']}**"
                f"  |  方向：**{overall_best['方向']}**\n\n"
                "⚠️ 回測基於歷史數據，未來不保證相同表現。請嚴格執行止損策略。"
            )
        else:
            st.info(f"三個維度均無 ≥ {_wr_thr}% 勝率組合。建議延長時間範圍至 **1y** 以上，"
                    "或降低高勝率閾值。")

        # ══════════════════════════════════════════════════════════════════════
        # 🔀 一鍵合併三維度到 Telegram 觸發條件（覆蓋模式）
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("🔀 一鍵合併三維度 → Telegram 觸發條件")
        st.caption(
            "將三個維度的回測結果依勝率閾值篩選後合併，"
            "**完整覆蓋**現有 Telegram 觸發條件表（不保留舊條件）。"
        )

        _merge_col1, _merge_col2, _merge_col3 = st.columns([1, 1, 3])

        _merge_thr = _merge_col1.number_input(
            "納入勝率閾值 (%)",
            min_value=0,
            max_value=100,
            value=_wr_thr,
            step=5,
            key="merge_thr",
            help="三個維度中勝率高於此值的組合才會被納入，設為 0 則全部納入",
        )

        # 預覽：計算三維合併後的條數
        def _preview_merge(thr: float):
            """回傳合併後（去重前）的預覽 DataFrame，格式與條件表一致。"""
            rows = []
            # 維度一：純信號組合
            for _, r in df_sig.iterrows():
                if r["勝率(%)"] >= thr:
                    rows.append({
                        "異動標記":   r["信號組合"].replace(" + ", ", "),
                        "成交量標記": "—",
                        "K線形態":    "—",
                        "回測勝率":   f"{r['勝率(%)']:.1f}%",
                        "方向":       r.get("方向", "做多"),
                        "_wr_num":    r["勝率(%)"],
                    })
            # 維度二：信號 + 成交量
            for _, r in df_vol.iterrows():
                if r["勝率(%)"] >= thr:
                    rows.append({
                        "異動標記":   r["信號組合"].replace(" + ", ", "),
                        "成交量標記": r.get("成交量標記", "—"),
                        "K線形態":    "—",
                        "回測勝率":   f"{r['勝率(%)']:.1f}%",
                        "方向":       r.get("方向", "做多"),
                        "_wr_num":    r["勝率(%)"],
                    })
            # 維度三：信號 + K線形態
            for _, r in df_kl.iterrows():
                if r["勝率(%)"] >= thr:
                    rows.append({
                        "異動標記":   r["信號組合"].replace(" + ", ", "),
                        "成交量標記": "—",
                        "K線形態":    r.get("K線形態", "—"),
                        "回測勝率":   f"{r['勝率(%)']:.1f}%",
                        "方向":       r.get("方向", "做多"),
                        "_wr_num":    r["勝率(%)"],
                    })
            if not rows:
                return pd.DataFrame()
            merged = (
                pd.DataFrame(rows)
                .sort_values("_wr_num", ascending=False)
                .drop_duplicates(subset=["異動標記","成交量標記","K線形態"], keep="first")
                .drop(columns=["_wr_num"])
                .reset_index(drop=True)
            )
            merged["排名"] = [str(i + 1) for i in range(len(merged))]
            if "方向" not in merged.columns:
                merged["方向"] = ""
            return merged[["排名","異動標記","成交量標記","K線形態","回測勝率","方向"]]

        _preview_df = _preview_merge(float(_merge_thr))
        _n_preview  = len(_preview_df)

        # 預覽數量提示
        with _merge_col2:
            st.metric(
                "合併後條數",
                f"{_n_preview} 條",
                help="去重後將覆蓋現有條件表的全部內容",
            )

        with _merge_col3:
            if _n_preview == 0:
                st.warning(
                    f"目前三個維度中無勝率 ≥ {_merge_thr}% 的組合，"
                    "請降低閾值或重新回測。"
                )
            else:
                st.info(
                    f"✅ 三個維度合併後共 **{_n_preview}** 條（去重後），"
                    f"將**完整覆蓋**現有 Telegram 條件表。\n"
                    "現有條件表中的所有舊條件將被清除。"
                )

        # 預覽表格（可展開）
        if not _preview_df.empty:
            with st.expander(f"👁️ 預覽合併結果（{_n_preview} 條，點擊展開）"):
                st.dataframe(
                    _preview_df.style.background_gradient(
                        subset=["回測勝率"],
                        cmap="Greens",
                        gmap=_preview_df["回測勝率"].str.replace("%","",regex=False).apply(pd.to_numeric,errors="coerce"),
                    ),
                    use_container_width=True,
                    height=min(500, 38 * (_n_preview + 1) + 40),
                    column_config={
                        "排名":       st.column_config.TextColumn("排名",       width="small"),
                        "異動標記":   st.column_config.TextColumn("異動標記",   width="large"),
                        "成交量標記": st.column_config.TextColumn("成交量標記", width="small"),
                        "K線形態":    st.column_config.TextColumn("K線形態",    width="medium"),
                        "回測勝率":   st.column_config.TextColumn("回測勝率",   width="small"),
                    },
                )

        # 執行按鈕
        _btn_disabled = (_n_preview == 0)
        if st.button(
            f"🔀 確認合併並覆蓋 Telegram 條件表（共 {_n_preview} 條）",
            type="primary",
            disabled=_btn_disabled,
            key="merge_all_dims_btn",
            help="此操作將清除現有條件表所有內容，以三維合併結果取代",
        ):
            # 執行覆蓋
            _merged_saved = _preview_df.copy()
            st.session_state["tg_conds"] = _merged_saved
            _tg_save(_merged_saved)
            _old_count = len(st.session_state.get("tg_conds", pd.DataFrame()))
            st.success(
                f"🎯 **覆蓋完成！** Telegram 觸發條件表已更新為三維合併結果。\n\n"
                f"共 **{_n_preview}** 條條件（勝率 ≥ {_merge_thr}%），"
                f"依勝率由高至低排名。\n\n"
                "📋 請捲動至頁面頂部的「**Telegram 觸發條件配置**」表格確認。\n"
                "系統每次刷新時，最新K線若符合其中任一條件即自動觸發 Telegram 交易信號。"
            )
            st.balloons()

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
