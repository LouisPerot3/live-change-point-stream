import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import joblib
import requests
import datetime
import time

MODEL_PATH = "ML_model.pkl"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1361352885786644672/UcHkWLhKJDHnbriFJCBTzmb5HshkJ2T-ZzWHQCwmN4Vxsx8BTlDNTImQpb1qFsoWxGoE"

def generate_features(returns: pd.Series, window: int = 60):
    if len(returns) < window:
        return None
    window_data = returns[-window:]
    s = pd.Series(window_data)
    feats = pd.DataFrame([{
        "mean": float(s.mean()),
        "std": float(s.std()),
        "skew": float(s.skew()),
        "kurt": float(s.kurt()),
        "min": float(s.min()),
        "max": float(s.max()),
        "last": float(s.iloc[-1]),
    }])
    return feats

def send_discord_alert(ticker, proba, direction):
    emoji = "⬆️" if direction == "hausse" else "⬇️"
    message = f"⚠️ Rupture sur `{ticker}` {emoji} - Probabilité: {proba:.2%}"
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
    except Exception as e:
        print("[Discord] Erreur alerte:", e)

def send_discord_start_message(tickers):
    if isinstance(tickers, str):
        tickers = [tickers]
    for t in tickers:
        message = f"🚀 Détection live lancée sur `{t}`"
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
        except Exception as e:
            print(f"[Discord] Erreur pour {t}:", e)

def fetch_returns(ticker="AAPL", period="7d", interval="1m"):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty or "Close" not in df.columns:
            st.error(f"⛔ Le ticker `{ticker}` est invalide ou ne retourne aucune donnée.")
            return None
        prices = df["Close"].dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        # 🔧 Correction clé ici :
        returns = pd.Series(returns.values.flatten(), index=prices.index[-len(returns):])
        print("✅ Shape des returns:", returns.shape)
        return returns
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du ticker `{ticker}` : {e}")
        return None

def run_dashboard():
    st.title("Multi-Ticker Live Change Point Detection")

    tickers_input = st.text_input("Tickers à surveiller (séparés par des virgules)", value="AAPL,MSFT")
    window = st.slider("Taille de la fenêtre (features)", 30, 120, 60)
    interval = st.slider("Fréquence de mise à jour (secondes)", 10, 60, 30)
    enable_alerts = st.checkbox("Activer les alertes Discord")

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if enable_alerts:
        send_discord_start_message(tickers)

    model = joblib.load(MODEL_PATH)
    placeholders = {ticker: st.empty() for ticker in tickers}
    last_status = {ticker: None for ticker in tickers}

        if st.button("🔄 Rafraîchir les données maintenant"):
        for ticker in tickers:
            data = fetch_returns(ticker)
            if data is None or len(data) < window:
                placeholders[ticker].warning(f"Pas de données valides pour {ticker}")
                continue

            X = generate_features(data, window)
            if X is None:
                placeholders[ticker].warning(f"Pas assez de données pour {ticker}")
                continue

            proba = model.predict_proba(X)[0][1]
            pred = model.predict(X)[0]

            if pred == 1:
                direction = "hausse" if X["mean"].iloc[0] > 0 else "baisse"
                label = f"Rupture probable ({direction})"
                couleur = "red"
                if enable_alerts and last_status[ticker] != 1:
                    send_discord_alert(ticker, proba, direction)
            else:
                label = f"Stabilité probable"
                couleur = "green"

            fig = make_subplots(specs=[[{"secondary_y": False}]])
            fig.add_trace(go.Scatter(x=data.index, y=data.values, name="Log-returns", line=dict(color=couleur)))
            fig.update_layout(title=f"{ticker} | {label} | P(Rupture): {proba:.2%}", height=400)

            placeholders[ticker].plotly_chart(fig, use_container_width=True)
            last_status[ticker] = pred

        st.caption(f"Mise à jour à {datetime.datetime.now().strftime('%H:%M:%S')}")
        time.sleep(interval)

if __name__ == "__main__":
    run_dashboard()
