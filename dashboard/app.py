import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import (
    get_connection,
    get_news_sentiment_by_date,
    get_social_sentiment_by_date,
    get_stock_prices_by_date,
    get_latest_prediction,
    get_all_tickers_summary
)
from collectors.stock_collector import get_all_latest_prices
from analysis.prediction import predict_tomorrow

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Market Intelligence Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .positive { color: #2ecc71; font-weight: bold; }
    .negative { color: #e74c3c; font-weight: bold; }
    .neutral  { color: #f39c12; font-weight: bold; }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

STOCKS = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon"
}

with st.sidebar:
    st.image("https://img.icons8.com/color/96/stock-share.png", width=60)
    st.title("Market Intelligence")
    st.divider()

    selected_ticker = st.selectbox(
        "Select Stock",
        options=list(STOCKS.keys()),
        format_func=lambda x: f"{x} — {STOCKS[x]}"
    )

    days = st.slider("Analysis Period (days)", 7, 30, 14)

    st.divider()
    st.caption("Data Sources")
    st.markdown("📰 NewsAPI + Google RSS")
    st.markdown("💬 StockTwits")
    st.markdown("📈 Yahoo Finance")
    st.markdown("🤖 FinBERT + XGBoost")

    st.divider()
    last_updated = datetime.now().strftime("%b %d, %Y %H:%M")
    st.caption(f"Last updated: {last_updated}")

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────

st.markdown(
    '<p class="main-header">📈 Market Intelligence Engine</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">'
    'Real-time sentiment analysis & next-day price prediction '
    'for top US stocks</p>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────
# ROW 1 — LIVE PRICES (ALL 5 STOCKS)
# ─────────────────────────────────────────

st.subheader("🔴 Live Prices")

try:
    live_prices = get_all_latest_prices()
    if live_prices:
        cols = st.columns(5)
        for i, price in enumerate(live_prices):
            with cols[i]:
                delta_color = (
                    "normal" if price["pct_change"] >= 0
                    else "inverse"
                )
                st.metric(
                    label=f"{price['ticker']}",
                    value=f"${price['current_price']}",
                    delta=f"{price['pct_change']:+.2f}%"
                )
except Exception as e:
    st.warning(f"Live prices unavailable: {e}")

st.divider()

# ─────────────────────────────────────────
# ROW 2 — SENTIMENT METRICS
# ─────────────────────────────────────────

st.subheader(f"📊 Sentiment Overview — {selected_ticker}")

conn = get_connection()

# Today's sentiment
today_query = """
    SELECT 
        AVG(sentiment_score)   as avg_sentiment,
        COUNT(*)               as total_articles,
        SUM(CASE WHEN sentiment = 'positive' 
            THEN 1 ELSE 0 END) as positive,
        SUM(CASE WHEN sentiment = 'negative' 
            THEN 1 ELSE 0 END) as negative,
        SUM(CASE WHEN sentiment = 'neutral'  
            THEN 1 ELSE 0 END) as neutral
    FROM news_articles
    WHERE ticker = ?
"""
today_df = pd.read_sql(today_query, conn, params=(selected_ticker,))

avg_score     = float(today_df["avg_sentiment"].iloc[0] or 0)
total         = int(today_df["total_articles"].iloc[0] or 0)
positive      = int(today_df["positive"].iloc[0] or 0)
negative      = int(today_df["negative"].iloc[0] or 0)
neutral_count = int(today_df["neutral"].iloc[0] or 0)

# Overall mood label
if avg_score > 0.1:
    mood       = "😊 Positive"
    mood_color = "positive"
elif avg_score < -0.1:
    mood       = "😠 Negative"
    mood_color = "negative"
else:
    mood       = "😐 Neutral"
    mood_color = "neutral"

# ML Prediction
pred_df = get_latest_prediction(selected_ticker)
if not pred_df.empty:
    prediction  = pred_df.iloc[0]["predicted_movement"]
    pred_conf   = round(float(pred_df.iloc[0]["confidence"]) * 100, 1)
    pred_label  = f"📈 {prediction}" if "UP" in str(prediction) \
                  else f"📉 {prediction}"
else:
    pred_label = "N/A"
    pred_conf  = 0

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Overall Mood",     mood)
col2.metric("Avg Sentiment",    f"{avg_score:.3f}")
col3.metric("Total Articles",   total)
col4.metric("Positive / Neg",   f"{positive} / {negative}")
col5.metric("ML Prediction",    pred_label,
            delta=f"{pred_conf}% confidence")

st.divider()

# ─────────────────────────────────────────
# ROW 3 — SENTIMENT VS PRICE CHART
# ─────────────────────────────────────────

st.subheader(f"📉 Sentiment vs Price — {selected_ticker}")

news_df  = get_news_sentiment_by_date(selected_ticker, days)
price_df = get_stock_prices_by_date(selected_ticker, days)

if not news_df.empty and not price_df.empty:
    merged = pd.merge(news_df, price_df, on="date", how="inner")
    merged = merged.sort_values("date")

    fig = go.Figure()

    # Sentiment bars
    fig.add_trace(go.Bar(
        x=merged["date"],
        y=merged["avg_sentiment"],
        name="News Sentiment",
        marker_color=[
            "#2ecc71" if x > 0 else "#e74c3c"
            for x in merged["avg_sentiment"]
        ],
        opacity=0.7,
        yaxis="y1"
    ))

    # Price line
    fig.add_trace(go.Scatter(
        x=merged["date"],
        y=merged["close_price"],
        name="Close Price",
        line=dict(color="#1f77b4", width=2.5),
        yaxis="y2"
    ))

    fig.update_layout(
        yaxis=dict(
            title=dict(
                text="Sentiment Score",
                font=dict(color="#e74c3c")
            )
        ),
        yaxis2=dict(
            title=dict(
                text="Stock Price ($)",
                font=dict(color="#1f77b4")
            ),
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0, y=1.1, orientation="h"),
        hovermode="x unified",
        height=400,
        margin=dict(l=40, r=40, t=20, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data yet — run collectors to gather more data")

# ─────────────────────────────────────────
# ROW 4 — SENTIMENT BREAKDOWN PIE
# ─────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🥧 Sentiment Breakdown")

    if total > 0:
        pie_fig = go.Figure(go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=[positive, negative, neutral_count],
            hole=0.4,
            marker_colors=["#2ecc71", "#e74c3c", "#f39c12"]
        ))
        pie_fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.info("No sentiment data yet")

with col_right:
    st.subheader("📊 Volume Trend")

    if not price_df.empty:
        price_df_sorted = price_df.sort_values("date")
        vol_fig = go.Figure(go.Bar(
            x=price_df_sorted["date"],
            y=price_df_sorted["volume"],
            marker_color="#1f77b4",
            opacity=0.7
        ))
        vol_fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=40),
            yaxis_title="Volume"
        )
        st.plotly_chart(vol_fig, use_container_width=True)
    else:
        st.info("No volume data yet")

st.divider()

# ─────────────────────────────────────────
# ROW 5 — ALL STOCKS HEATMAP
# ─────────────────────────────────────────

st.subheader("🌡️ Sentiment Heatmap — All Stocks")

try:
    heatmap_data = {}
    for ticker in STOCKS:
        ticker_df = get_news_sentiment_by_date(ticker, days)
        if not ticker_df.empty:
            ticker_df = ticker_df.set_index("date")
            heatmap_data[ticker] = ticker_df["avg_sentiment"]

    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data).fillna(0)
        heatmap_df = heatmap_df.sort_index()

        heat_fig = px.imshow(
            heatmap_df.T,
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            aspect="auto",
            labels=dict(
                x="Date",
                y="Stock",
                color="Sentiment"
            )
        )
        heat_fig.update_layout(
            height=250,
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(heat_fig, use_container_width=True)
    else:
        st.info("Not enough data for heatmap yet")
except Exception as e:
    st.warning(f"Heatmap unavailable: {e}")

st.divider()

# ─────────────────────────────────────────
# ROW 6 — TOMORROW'S PREDICTIONS
# ─────────────────────────────────────────

st.subheader("🔮 Tomorrow's Predictions (ML Model)")

pred_cols = st.columns(5)

for i, (ticker, company) in enumerate(STOCKS.items()):
    with pred_cols[i]:
        pred = predict_tomorrow(ticker)
        if pred:
            arrow = "📈" if pred["direction"] == 1 else "📉"
            color = (
                "green" if pred["direction"] == 1
                else "red"
            )
            st.markdown(
                f"**{ticker}**<br>"
                f"<span style='color:{color};font-size:1.5rem'>"
                f"{arrow}</span><br>"
                f"<small>{pred['confidence']}% confidence</small>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"**{ticker}**<br>N/A")

st.divider()

# ─────────────────────────────────────────
# ROW 7 — LATEST HEADLINES
# ─────────────────────────────────────────

st.subheader(f"🗞️ Latest Headlines — {selected_ticker}")

headlines_query = """
    SELECT title, source, sentiment, confidence, published_at
    FROM news_articles
    WHERE ticker = ?
    ORDER BY published_at DESC
    LIMIT 10
"""
headlines_df = pd.read_sql(
    headlines_query, conn,
    params=(selected_ticker,)
)

if not headlines_df.empty:
    for _, row in headlines_df.iterrows():
        emoji = (
            "🟢" if row["sentiment"] == "positive"
            else "🔴" if row["sentiment"] == "negative"
            else "🟡"
        )
        conf = round(float(row["confidence"] or 0) * 100, 0)
        st.markdown(
            f"{emoji} **{row['title']}**  \n"
            f"<small>📰 {row['source']} &nbsp;|&nbsp; "
            f"{row['sentiment'].upper()} ({conf:.0f}%) &nbsp;|&nbsp; "
            f"{row['published_at'][:10]}</small>",
            unsafe_allow_html=True
        )
        st.markdown("---")
else:
    st.info("No headlines yet — run news collector first")

# ─────────────────────────────────────────
# ROW 8 — LATEST DIGEST
# ─────────────────────────────────────────

st.subheader("📋 Latest Market Digest")

digest_query = """
    SELECT content, report_date, generated_at
    FROM daily_digests
    ORDER BY generated_at DESC
    LIMIT 1
"""
digest_df = pd.read_sql(digest_query, conn)

if not digest_df.empty:
    st.markdown(
        f"*Generated: {digest_df.iloc[0]['generated_at']}*"
    )
    st.text_area(
        label="",
        value=digest_df.iloc[0]["content"],
        height=300
    )
else:
    st.info("No digest yet — run digest generator first")

conn.close()

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────

st.divider()
st.markdown(
    "<center><small>"
    "Market Intelligence Engine | "
    "Built with FinBERT + XGBoost + Streamlit | "
    "Data: NewsAPI · StockTwits · Yahoo Finance"
    "</small></center>",
    unsafe_allow_html=True
)