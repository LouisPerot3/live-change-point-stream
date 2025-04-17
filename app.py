import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import joblib
import requests
import datetime
from streamlit_autorefresh import st_autorefresh

# ====== PARAMS ======
MODEL_PATH = "xgb_model_v5.pkl"
SCALER_PATH = "scaler_v5.pkl"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1361352885786644672/UcHkWLhKJDHnbriFJCBTzmb5HshkJ2T-ZzWHQCwmN4Vxsx8BTlDNTImQpb1qFsoWxGoE"
scaler = joblib.load(SCALER_PATH)
# ====== FEATURES ======
def generate_features(returns: pd.Series, window: int = 60):
    if len(returns) < window:
        return None
    s = pd.Series(returns[-window:].values.flatten())
    feats = np.array([[
        s.mean(),
        s.std(),
        s.skew(),
        s.kurt(),
        s.min(),
        s.max(),
        s.iloc[-1],
    ]])
    return feats

# ====== DISCORD ======
def send_discord_alert(ticker, proba, direction):
    emoji = "⬆️" if direction == "hausse" else "⬇️"
    message = f"\u26a0\ufe0f Rupture sur `{ticker}` {emoji} - Probabilité: {proba:.2%}"
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
    except Exception as e:
        print("[Discord] Erreur:", e)

def send_discord_start_message(tickers):
    if isinstance(tickers, str): tickers = [tickers]
    for t in tickers:
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": f"🚀 Détection live lancée sur `{t}`"})
        except: pass

# ====== FETCH RETURNS ======
def fetch_returns(ticker="AAPL", period="7d", interval="1m"):
    df = yf.download(ticker, period=period, interval=interval)
    if df.empty or "Close" not in df.columns:
        return None
    prices = df["Close"].dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return pd.Series(returns.values.flatten(), index=prices.index[-len(returns):])

# ====== STREAMLIT DASHBOARD ======
def run_dashboard():
    st_autorefresh(interval=30000, key="refresh")  # auto-refresh every 30s

    st.title("🌎 Live Change Point Detection - Cloud Mode")

    tickers_input = st.text_input("Tickers à surveiller (ex: AAPL, BTC-USD)", value="AAPL,MSFT")
    window = st.slider("Fenêtre d'analyse (features)", 30, 120, 60)
    enable_alerts = st.checkbox("🚨 Alertes Discord", value=True)

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    model = joblib.load(MODEL_PATH)

    placeholders = {t: st.empty() for t in tickers}
    last_status = st.session_state.get("last_status", {t: None for t in tickers})

    if enable_alerts and not st.session_state.get("startup_msg_sent", False):
        send_discord_start_message(tickers)
        st.session_state["startup_msg_sent"] = True

    for ticker in tickers:
        returns = fetch_returns(ticker)
        if returns is None or len(returns) < window:
            placeholders[ticker].warning(f"Pas de données valides pour {ticker}")
            continue

        X = generate_features(returns, window)
        if X is None:
            placeholders[ticker].warning(f"Données insuffisantes pour {ticker}")
            continue
        X = scaler.transform(X)
        proba = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]
        direction = "hausse" if X[0][0] > 0 else "baisse"

        if pred == 1:
            label = f"Rupture probable ({direction})"
            couleur = "red"
            if enable_alerts and last_status.get(ticker) != 1:
                send_discord_alert(ticker, proba, direction)
        else:
            label = "Stabilité probable"
            couleur = "green"

        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(go.Scatter(x=returns.index, y=returns.values, name="Log-returns", line=dict(color=couleur)))
        fig.update_layout(title=f"{ticker} | {label} | P(Rupture): {proba:.2%}", height=400)

        placeholders[ticker].plotly_chart(fig, use_container_width=True)
        last_status[ticker] = pred

    st.session_state["last_status"] = last_status
    st.caption(f"🔄 Mise à jour : {datetime.datetime.now().strftime('%H:%M:%S')} - toutes les 30s")

if __name__ == "__main__":
    run_dashboard()
