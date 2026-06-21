import sqlite3
import pandas as pd
import requests
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import get_connection, save_digest

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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

OUTPUT_DIR = "data/processed/digests"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_yesterday_summary():
    conn      = get_connection()
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    summary   = {}

    for ticker, company in STOCKS.items():
        news_query = """
            SELECT
                COUNT(*)                as total_articles,
                AVG(sentiment_score)    as avg_sentiment,
                SUM(CASE WHEN sentiment = 'positive'
                    THEN 1 ELSE 0 END)  as positive_count,
                SUM(CASE WHEN sentiment = 'negative'
                    THEN 1 ELSE 0 END)  as negative_count
            FROM news_articles
            WHERE ticker = ?
            AND DATE(fetched_at) >= ?
        """
        news_df = pd.read_sql(news_query, conn, params=(ticker, yesterday))

        social_query = """
            SELECT
                COUNT(*)                as total_posts,
                AVG(sentiment_score)    as avg_sentiment,
                SUM(CASE WHEN user_sentiment = 'Bullish'
                    THEN 1 ELSE 0 END)  as bullish_count,
                SUM(CASE WHEN user_sentiment = 'Bearish'
                    THEN 1 ELSE 0 END)  as bearish_count
            FROM social_posts
            WHERE ticker = ?
            AND DATE(fetched_at) >= ?
        """
        social_df = pd.read_sql(social_query, conn, params=(ticker, yesterday))

        price_query = """
            SELECT close_price, volume, date
            FROM stock_prices
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 2
        """
        price_df = pd.read_sql(price_query, conn, params=(ticker,))

        price_change_pct = None
        if len(price_df) >= 2:
            latest = price_df.iloc[0]["close_price"]
            prev   = price_df.iloc[1]["close_price"]
            price_change_pct = round(((latest - prev) / prev) * 100, 2)

        pred_query = """
            SELECT predicted_movement, confidence
            FROM predictions
            WHERE ticker = ?
            ORDER BY predicted_at DESC
            LIMIT 1
        """
        pred_df = pd.read_sql(pred_query, conn, params=(ticker,))

        summary[ticker] = {
            "company":          company,
            "ticker":           ticker,
            "total_articles":   int(news_df["total_articles"].iloc[0] or 0),
            "news_sentiment":   round(float(news_df["avg_sentiment"].iloc[0] or 0), 3),
            "positive_news":    int(news_df["positive_count"].iloc[0] or 0),
            "negative_news":    int(news_df["negative_count"].iloc[0] or 0),
            "total_posts":      int(social_df["total_posts"].iloc[0] or 0),
            "social_sentiment": round(float(social_df["avg_sentiment"].iloc[0] or 0), 3),
            "bullish_posts":    int(social_df["bullish_count"].iloc[0] or 0),
            "bearish_posts":    int(social_df["bearish_count"].iloc[0] or 0),
            "latest_price":     round(float(price_df.iloc[0]["close_price"]), 2)
                                if not price_df.empty else None,
            "price_change_pct": price_change_pct,
            "prediction":       pred_df.iloc[0]["predicted_movement"]
                                if not pred_df.empty else "N/A",
            "pred_confidence":  round(float(pred_df.iloc[0]["confidence"]) * 100, 1)
                                if not pred_df.empty else None
        }

    conn.close()
    return summary


def build_prompt(summary):
    stocks_text = ""
    for ticker, data in summary.items():
        sentiment_label = (
            "POSITIVE" if data["news_sentiment"] > 0.1
            else "NEGATIVE" if data["news_sentiment"] < -0.1
            else "NEUTRAL"
        )
        price_arrow = "▲" if (data["price_change_pct"] or 0) > 0 else "▼"

        stocks_text += f"""
{data['company']} ({ticker}):
- Sentiment: {sentiment_label} ({data['news_sentiment']:.2f})
- News: {data['total_articles']} articles | Positive: {data['positive_news']} | Negative: {data['negative_news']}
- Social: {data['total_posts']} posts | Bullish: {data['bullish_posts']} | Bearish: {data['bearish_posts']}
- Price: ${data['latest_price']} {price_arrow} {data['price_change_pct']}%
- ML Prediction: {data['prediction']} ({data['pred_confidence']}% confidence)
"""

    prompt = f"""
You are a professional financial analyst writing a morning market briefing.
Today's Date: {datetime.now().strftime("%B %d, %Y")}

DATA:
{stocks_text}

Write the briefing in this format:

MARKET INTELLIGENCE BRIEFING
{datetime.now().strftime("%B %d, %Y")}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MARKET OVERVIEW
[2-3 sentences summarizing overall sentiment]

STOCK ANALYSIS
[For each stock: 2 sentences on sentiment and prediction]

KEY SIGNALS TO WATCH
[3 bullet points]

Keep it under 400 words. Do not make buy/sell recommendations.
"""
    return prompt


def generate_digest():
    print("\n📝 Generating Daily Market Digest...")
    print("-" * 40)

    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in environment")
        return None

    print("📊 Gathering data...")
    summary = get_yesterday_summary()
    prompt  = build_prompt(summary)

    print("🤖 Calling Groq API directly...")

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role":    "system",
                    "content": "You are a professional financial analyst. Write factual market briefings. Never give investment advice."
                },
                {
                    "role":    "user",
                    "content": prompt
                }
            ],
            "max_tokens":  800,
            "temperature": 0.3
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
