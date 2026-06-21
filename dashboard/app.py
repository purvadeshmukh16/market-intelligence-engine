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
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }

    /* Cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }

    /* Headers */
    h1, h2, h3 {
        color: #58a6ff !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #e6edf3 !important;
    }

    /* Divider */
    hr {
        border-color: #30363d !important;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Main header */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .sub-title {
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Prediction cards */
    .pred-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# STOCKS CONFIG
# ─────────────────────────────────────────

STOCKS = {
    "AAPL":        "Apple",
    "TSLA":        "Tesla",
    "MSFT":        "Microsoft",
    "NVDA":        "NVIDIA",
    "AMZN":        "Amazon",
    "GOOGL":       "Google",
    "META":        "Meta",
    "RELIANCE.NS": "Reliance",
    "TCS.NS":      "TCS"
}

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Market Intelligence")
    st.markdown("*AI-powered stock analysis*")
    st.divider()

    selected_ticker = st.selectbox(
        "🔍 Select Stock",
        options=list(STOCKS.keys()),
        format_func=lambda x: f"{x} — {STOCKS[x]}"
    )

    days = st.slider("📅 Analysis Period (days)", 7, 30, 14)

    st.divider()
    st.markdown("**📡 Data Sources**")
    st.markdown("📰 NewsAPI + Google RSS")
    st.markdown("💬 StockTwits")
    st.markdown("📈 Yahoo Finance")
    st.markdown("🤖 FinBERT + XGBoost")

    st.divider()
    st.markdown("**🔄 Pipeline**")
    st.markdown("⚡ GitHub Actions CI/CD")
    st.markdown("🕕 Auto-runs at 6 AM daily")

    st.divider()
    last_updated = datetime.now().strftime("%b %d, %Y %H:%M")
    st.caption(f"🕐 Last updated: {last_updated}")

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────

st.markdown(
    '<p class="main-title">📈 Market Intelligence Engine</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-title">Real-time sentiment analysis & '
    'next-day price prediction powered by FinBERT + XGBoost</p>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────
# ROW 1 — LIVE PRICES
# ─────────────────────────────────────────

st.subheader("🔴 Live Prices")

try:
    live_prices = get_all_latest_prices()
    if live_prices:
        # Show 5 per row
        row1 = live_prices[:5]
        row2 = live_prices[5:] if len(live_prices) > 5 else []

        cols1 = st.columns(len(row1))
        for i, price in enumerate(row1):
            with cols1[i]:
                st.metric(
                    label=f"**{price['ticker']}**",
                    value=f"${price['current_price']}",
                    delta=f"{price['pct_change']:+.2f}%"
                )

        if row2:
            cols2 = st.columns(len(row2))
            for i, price in enumerate(row2):
                with cols2[i]:
                    st.metric(
                        label=f"**{price['ticker']}**",
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

if avg_score > 0.1:
    mood = "😊 Positive"
elif avg_score < -0.1:
    mood = "😠 Negative"
else:
    mood = "😐 Neutral"

pred_df = get_latest_prediction(selected_ticker)
if not pred_df.empty:
    prediction = pred_df.iloc[0]["predicted_movement"]
    pred_conf  = round(float(pred_df.iloc[0]["confidence"]) * 100, 1)
    pred_label = f"📈 {prediction}" if "UP" in str(prediction) \
                 else f"📉 {prediction}"
else:
    pred_label = "N/A"
    pred_conf  = 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Overall Mood",   mood)
col2.metric("Avg Sentiment",  f"{avg_score:.3f}")
col3.metric("Total Articles", total)
col4.metric("Pos / Neg",      f"{positive} / {negative}")
col5.metric("ML Prediction",  pred_label,
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
    fig.add_trace(go.Scatter(
        x=merged["date"],
        y=merged["close_price"],
        name="Close Price",
        line=dict(color="#58a6ff", width=2.5),
        yaxis="y2"
    ))
    fig.update_layout(
        yaxis=dict(
            title=dict(text="Sentiment Score",
                      font=dict(color="#e74c3c"))
        ),
        yaxis2=dict(
            title=dict(text="Stock Price ($)",
                      font=dict(color="#58a6ff")),
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0, y=1.1, orientation="h"),
        hovermode="x unified",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        margin=dict(l=40, r=40, t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data yet — run collectors to gather more data")

# ─────────────────────────────────────────
# ROW 4 — PIE + VOLUME
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
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.info("No sentiment data yet")

with col_right:
    st.subheader("📊 Volume Trend")
    if not price_df.empty:
        price_sorted = price_df.sort_values("date")
        vol_fig = go.Figure(go.Bar(
            x=price_sorted["date"],
            y=price_sorted["volume"],
            marker_color="#58a6ff",
            opacity=0.7
        ))
        vol_fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            margin=dict(l=20, r=20, t=20, b=40),
            yaxis=dict(title=dict(text="Volume"))
        )
        st.plotly_chart(vol_fig, use_container_width=True)
    else:
        st.info("No volume data yet")

st.divider()

# ─────────────────────────────────────────
# ROW 5 — HEATMAP
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
            labels=dict(x="Date", y="Stock", color="Sentiment")
        )
        heat_fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(heat_fig, use_container_width=True)
    else:
        st.info("Not enough data for heatmap yet")
except Exception as e:
    st.warning(f"Heatmap unavailable: {e}")

st.divider()

# ─────────────────────────────────────────
# ROW 6 — PREDICTIONS (FIXED FOR 9 STOCKS)
# ─────────────────────────────────────────

st.subheader("🔮 Tomorrow's Predictions (ML Model)")

stock_list  = list(STOCKS.items())
row1_stocks = stock_list[:5]
row2_stocks = stock_list[5:]

def show_prediction_cols(stocks_subset):
    cols = st.columns(len(stocks_subset))
    for i, (ticker, company) in enumerate(stocks_subset):
        with cols[i]:
            try:
                pred = predict_tomorrow(ticker)
                if pred and isinstance(pred, dict):
                    direction = pred.get("direction", None)
                    confidence = pred.get("confidence", 0)

                    if direction == 1:
                        arrow = "📈"
                        color = "green"
                        move  = "UP"
                    elif direction == 0:
                        arrow = "📉"
                        color = "red"
                        move  = "DOWN"
                    else:
                        arrow = "⬜"
                        color = "gray"
                        move  = "N/A"

                    st.markdown(
                        f"<div style='text-align:center'>"
                        f"<b>{ticker}</b><br>"
                        f"<span style='font-size:2rem'>{arrow}</span><br>"
                        f"<span style='color:{color};"
                        f"font-weight:bold'>{move}</span><br>"
                        f"<small style='color:gray'>{company}</small><br>"
                        f"<small>{round(confidence)}% confidence</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='text-align:center'>"
                        f"<b>{ticker}</b><br>"
                        f"<span style='font-size:2rem'>⬜</span><br>"
                        f"<span style='color:gray'>N/A</span><br>"
                        f"<small style='color:gray'>{company}</small><br>"
                        f"<small>No data yet</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<b>{ticker}</b><br>"
                    f"<span style='font-size:2rem'>⬜</span><br>"
                    f"<span style='color:gray'>N/A</span><br>"
                    f"<small style='color:gray'>{company}</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )

show_prediction_cols(row1_stocks)
st.markdown("<br>", unsafe_allow_html=True)
show_prediction_cols(row2_stocks)

st.divider()

# ─────────────────────────────────────────
# ROW 7 — HEADLINES
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
            f"{str(row['sentiment']).upper()} ({conf:.0f}%)"
            f" &nbsp;|&nbsp; "
            f"{str(row['published_at'])[:10]}</small>",
            unsafe_allow_html=True
        )
        st.markdown("---")
else:
    st.info("No headlines yet")

# ─────────────────────────────────────────
# ROW 8 — DIGEST
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
    "<center><small style='color:#8b949e'>"
    "📈 Market Intelligence Engine &nbsp;|&nbsp; "
    "Built with FinBERT + XGBoost + Streamlit &nbsp;|&nbsp; "
    "Data: NewsAPI · StockTwits · Yahoo Finance &nbsp;|&nbsp; "
    "Automated via GitHub Actions CI/CD"
    "</small></center>",
    unsafe_allow_html=True
)
