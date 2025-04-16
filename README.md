# Live Stream Change Point Detector

Cette app utilise un modèle LightGBM pour détecter en temps réel les ruptures de régime sur des actifs financiers (AAPL, BTC, etc.) via Streamlit et yFinance.

## ✅ Fonctionnalités

- Multi-ticker en streaming
- Affichage en temps réel avec courbes
- Alerte Discord en cas de rupture détectée
- Probabilité et direction de la rupture

## 🚀 Lancer en local

```bash
streamlit run streamlit_app.py
```

## 🌐 Déploiement sur Streamlit Cloud

1. Uploade ce repo sur GitHub
2. Va sur [streamlit.io/cloud](https://streamlit.io/cloud)
3. Clique "New app", choisis `streamlit_app.py` et déploie 🎯
