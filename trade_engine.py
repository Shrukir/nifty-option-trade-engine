# 🍆 Cell 1: Install and Import
!pip install beautifulsoup4 lxml scipy --quiet

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from datetime import datetime
from IPython.display import display
from scipy.stats import norm
import requests
import json
from math import log, sqrt, exp

# 🌐 Cell 2: Fetch Option Chain using Requests
def fetch_nifty_chain():
    try:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain",
        })

        _ = session.get("https://www.nseindia.com", timeout=5)
        response = session.get(url, timeout=5)

        if response.status_code != 200:
            print(f"❌ NSE fetch failed: HTTP {response.status_code}")
            return None

        data = response.json()
        records = data['records']['data']
        spot = data['records']['underlyingValue']

        rows = []
        for item in records:
            strike = item.get('strikePrice')
            for opt in ['CE', 'PE']:
                if opt in item:
                    row = item[opt]
                    row['Type'] = opt
                    row['Strike'] = strike
                    row['underlyingValue'] = spot
                    rows.append(row)

        df = pd.DataFrame(rows)
        df["LTP"] = pd.to_numeric(df["lastPrice"], errors="coerce")
        df["IV"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
        df["OI"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["Chg OI"] = pd.to_numeric(df["changeinOpenInterest"], errors="coerce")
        df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
        df["expiryDate"] = pd.to_datetime(df["expiryDate"], errors="coerce")
        df["Theta"] = df["LTP"].apply(lambda x: -abs(x) * 10 / 7 if pd.notnull(x) else 0)
        return df.dropna()

    except Exception as e:
        print("❌ NSE fetch failed:", e)
        return None

# 📊 Cell 3: Load Previous Day Data (Memory)
def load_previous_day_oi():
    try:
        df = pd.read_csv("previous_day_chain.csv")
        memory = {}
        for _, row in df.iterrows():
            key = f"{row['Type']}_{row['Strike']}"
            memory[key] = row['OI']
        return memory
    except Exception as e:
        print("❌ Error loading previous day data:", e)
        return {}

# 📌 Cell 4: ATM Strike Calculation
def get_atm_strike(df):
    spot = df['underlyingValue'].iloc[0]
    atm = round(spot / 50) * 50
    return spot, atm

# 🔍 Cell 5: Filter Data for Trade Suggestion
def filter_options(df, option_type, atm_strike):
    df = df[(df["Type"] == option_type) &
            (df["Strike"].between(atm_strike - 300, atm_strike + 300)) &
            (df["LTP"] > 10) &
            (df["IV"] > 0)]

    return df.sort_values("Chg OI", ascending=False).head(3)

# 🖐 BSM Delta Calculator
def compute_bsm_delta(S, K, T, r, sigma, option_type):
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        if option_type == "CE":
            return norm.cdf(d1)
        else:
            return -norm.cdf(-d1)
    except:
        return 0

# 📊 Cell 6: PCR and Volume Profile

def compute_pcr_and_volume_profile(df):
    df_pcr = df.groupby("Strike").apply(
        lambda x: x[x.Type == 'PE']['OI'].sum() / x[x.Type == 'CE']['OI'].sum() if not x[x.Type == 'CE'].empty else 0
    )
    df_vol = df.groupby("Strike")["OI"].sum()
    return df_pcr, df_vol

# 📛 Detect Gamma Blast Candidates
def detect_gamma_blast(df, atm):
    near_atm = df[df["Strike"].between(atm - 50, atm + 50)]
    high_theta = near_atm["Theta"].abs() > 100
    high_oi = near_atm["OI"] > near_atm["OI"].quantile(0.75)
    if (high_theta & high_oi).any():
        return True
    return False

# 🧠 Macro + Micro Signal Engine
macro_sentiment = {
    "SGX_Nifty": 50,  # dummy
    "India_VIX": 13.1,
    "Crude_Oil": 84,
    "USD_INR": 83.5,
    "FII_Net": -2800,  # crores
    "Event": "Fed Meet Tonight"
}

def get_market_tone(data):
    score = 0
    score += 1 if data['SGX_Nifty'] > 0 else -1
    score += -1 if data['India_VIX'] > 15 else 1
    score += -1 if data['Crude_Oil'] > 90 else 0
    score += -1 if data['USD_INR'] > 83.3 else 1
    score += 1 if data['FII_Net'] > 0 else -1

    if score >= 3:
        return "🟢 Bullish"
    elif score <= -2:
        return "🔴 Bearish"
    else:
        return "🟡 Sideways"

# 💡 Trade Suggestion Logic

def suggest_trades(df, atm_strike, memory):
    df["Key"] = df.apply(lambda row: f"{row['Type']}_{row['Strike']}", axis=1)
    df["Prev OI"] = df["Key"].apply(lambda x: memory.get(x, 0))
    df["OI Change %"] = np.where(df["Prev OI"] > 0,
                                  ((df["OI"] - df["Prev OI"]) / df["Prev OI"]) * 100,
                                  (df["Chg OI"] / df["OI"]) * 100)

    S = df['underlyingValue'].iloc[0]
    T = 3 / 365
    r = 0.06
    df["Delta"] = df.apply(lambda row: compute_bsm_delta(S, row["Strike"], T, r, row["IV"] / 100, row["Type"]), axis=1)

    def assign_confidence(row):
        if abs(row["Delta"]) > 0.5 and row["OI Change %"] > 30 and row["Theta"] > -100:
            return "🟢"
        elif abs(row["Delta"]) > 0.3 and row["OI Change %"] > 15:
            return "🟡"
        else:
            return "🔴"

    def trade_rating(row):
        score = 0
        score += min(max(abs(row["Delta"]) * 10, 0), 3.5)
        score += min(max(row["OI Change %"] / 10, 0), 4)
        score += max(0, 2.5 - abs(row["Theta"]) / 100)
        return round(min(score, 10), 1)

    df["Confidence"] = df.apply(assign_confidence, axis=1)
    df["Rating"] = df.apply(trade_rating, axis=1)
    df = df.sort_values("Rating", ascending=False)
    return df

# 🧮 Exit logic
def should_exit_trade(entry_price, current_ltp, theta):
    if current_ltp < 0.5 * entry_price:
        return "❌ Stop Loss Hit"
    elif theta < -100:
        return "⌛ High Theta Decay"
    elif current_ltp > 1.5 * entry_price:
        return "💰 Book Profit ✅"
    return "⏳ Hold"

# 🚀 Runtime Execution
df = fetch_nifty_chain()
if df is not None and not df.empty:
    market_tone = get_market_tone(macro_sentiment)
    event_today = macro_sentiment['Event']
    print(f"📊 Today’s Market Tone: {market_tone} | Event Risk: {event_today}")

    memory = load_previous_day_oi()
    spot_price, atm_strike = get_atm_strike(df)
    print(f"📍 Spot: {spot_price:.2f}, ATM Strike: {atm_strike}")

    df_pcr, df_vol = compute_pcr_and_volume_profile(df)
    if detect_gamma_blast(df, atm_strike):
        print("⚠️ Potential Gamma Blast setup near ATM")

    ce_df = filter_options(df, "CE", atm_strike)
    pe_df = filter_options(df, "PE", atm_strike)

    ce_trades = suggest_trades(ce_df, atm_strike, memory)
    pe_trades = suggest_trades(pe_df, atm_strike, memory)

    print("\n📈 Suggested Call Option Trades:")
    display(ce_trades[["Strike", "Type", "LTP", "IV", "OI", "Chg OI", "Delta", "Theta", "Confidence", "Rating"]])

    print("\n📉 Suggested Put Option Trades:")
    display(pe_trades[["Strike", "Type", "LTP", "IV", "OI", "Chg OI", "Delta", "Theta", "Confidence", "Rating"]])

    entry_price = 100

    print("\n📊 CE Greeks + Exit Suggestions:")
    for _, row in ce_trades.iterrows():
        print(f"CE {row['Strike']} | Δ: {row['Delta']:.3f}, Θ: {row['Theta']:.2f}, OIΔ%: {row['OI Change %']:.2f} | Confidence: {row['Confidence']} | Rating: {row['Rating']} | Exit: {should_exit_trade(entry_price, row['LTP'], row['Theta'])}")

    print("\n📊 PE Greeks + Exit Suggestions:")
    for _, row in pe_trades.iterrows():
        print(f"PE {row['Strike']} | Δ: {row['Delta']:.3f}, Θ: {row['Theta']:.2f}, OIΔ%: {row['OI Change %']:.2f} | Confidence: {row['Confidence']} | Rating: {row['Rating']} | Exit: {should_exit_trade(entry_price, row['LTP'], row['Theta'])}")

    os.makedirs("logs", exist_ok=True)
    today = datetime.today().strftime('%Y-%m-%d')
    ce_trades.to_csv("previous_day_chain.csv", index=False)
    ce_trades.to_csv(f"logs/{today}_ce.csv", index=False)
else:
    print("❌ Live data fetch failed. Retry after a few seconds.")
