# 📈 Market Intelligence Engine
> An automated real-time stock sentiment analysis and next-day 
> price prediction system powered by FinBERT NLP and XGBoost ML

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![FinBERT](https://img.shields.io/badge/NLP-FinBERT-green?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?style=for-the-badge)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-black?style=for-the-badge&logo=githubactions)

## 🔗 Live Demo

[View Live Dashboard](https://market-intelligence-engine-mudjcnqggxkngnscdvopol.streamlit.app/)

## 📌 Problem Statement
Financial markets are heavily influenced by public sentiment expressed through news and social media. Traditional analysis 
tools focus only on price data — ignoring the massive signal hidden in text.
This project builds a fully automated market intelligence system that collects live data, scores sentiment using AI, predicts next-day stock movement using ML, and auto-generates 
a professional morning briefing — all without any manual effort.

## ✨ Key Features
- Real-time Data Pipeline — Collects live news from NewsAPI + Google RSS + StockTwits every 2 hours for 9 stocks
- FinBERT Sentiment Analysis — Finance-specific BERT model scores every headline as positive/negative/neutral with 
  confidence scores
- ML Prediction System — Random Forest, XGBoost, and Logistic Regression trained to predict next-day price 
  direction (Up/Down)
- SHAP Explainability — Shows which features drive each prediction
- Correlation Analysis — Pearson correlation between sentiment and price movements with lag analysis
- LLM Daily Digest — Auto-generated professional morning market briefing using LLaMA 3 via Groq API
- Interactive Dashboard — Live Streamlit dashboard with sentiment heatmap, price charts, predictions and headlines
- Fully Automated CI/CD — GitHub Actions runs the entire pipeline every day at 6 AM UTC automatically

## Stocks Tracked
| Ticker | Company | Market |
|--------|---------|--------|
| AAPL | Apple Inc | US |
| TSLA | Tesla Inc | US |
| MSFT | Microsoft Corp | US |
| NVDA | NVIDIA Corp | US |
| AMZN | Amazon.com Inc | US |
| GOOGL | Alphabet (Google) | US |
| META | Meta Platforms | US |
| RELIANCE.NS | Reliance Industries | India |
| TCS.NS | Tata Consultancy Services | India |

## ⚠️ Disclaimer
This project is built for ""educational purposes only"". Nothing in this project constitutes financial or investment advice.  
Never make investment decisions based on this tool.
