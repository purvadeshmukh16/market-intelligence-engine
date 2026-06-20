import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import get_connection, save_digest

# ─────────────────────────────────────────
# CONFIG
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

OUTPUT_DIR = "data/processed/digests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# DATA GATHERING
# ─────────────────────────────────────────

def get_yesterday_summary():
    """
    Pull yesterday's sentiment + price data
    from DB for all tickers
    """
    conn      = get_connection()
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    today     = datetime.now().strftime("%Y-%m-%d")

    summary = {}

    for ticker, company in STOCKS.items():

        # News sentiment
        news_query = """
            SELECT 
                COUNT(*)                as total_articles,
                AVG(sentiment_score)    as avg_sentiment,
                SUM(CASE WHEN sentiment = 'positive' 
                    THEN 1 ELSE 0 END)  as positive_count,
                SUM(CASE WHEN sentiment = 'negative' 
                    THEN 1 ELSE 0 END)  as negative_count,
                SUM(CASE WHEN sentiment = 'neutral'  
                    THEN 1 ELSE 0 END)  as neutral_count
            FROM news_articles
            WHERE ticker = ?
            AND DATE(fetched_at) >= ?
        """
        news_df = pd.read_sql(
            news_query, conn,
            params=(ticker, yesterday)
        )

        # Social sentiment
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
        social_df = pd.read_sql(
            social_query, conn,
            params=(ticker, yesterday)
        )

        # Latest stock price
        price_query = """
            SELECT 
                close_price,
                volume,
                date
            FROM stock_prices
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 2
        """
        price_df = pd.read_sql(
            price_query, conn,
            params=(ticker,)
        )

        # Latest prediction
        pred_query = """
            SELECT 
                predicted_movement,
                confidence
            FROM predictions
            WHERE ticker = ?
            ORDER BY predicted_at DESC
            LIMIT 1
        """
        pred_df = pd.read_sql(
            pred_query, conn,
            params=(ticker,)
        )

        # Calculate price change
        price_change = None
        price_change_pct = None
        if len(price_df) >= 2:
            latest_price = price_df.iloc[0]["close_price"]
            prev_price   = price_df.iloc[1]["close_price"]
            price_change = round(latest_price - prev_price, 2)
            price_change_pct = round(
                ((latest_price - prev_price) / prev_price) * 100, 2
            )

        # Build summary dict for this ticker
        summary[ticker] = {
            "company":           company,
            "ticker":            ticker,

            # News data
            "total_articles":    int(news_df["total_articles"].iloc[0] or 0),
            "news_sentiment":    round(float(news_df["avg_sentiment"].iloc[0] or 0), 3),
            "positive_news":     int(news_df["positive_count"].iloc[0] or 0),
            "negative_news":     int(news_df["negative_count"].iloc[0] or 0),
            "neutral_news":      int(news_df["neutral_count"].iloc[0] or 0),

            # Social data
            "total_posts":       int(social_df["total_posts"].iloc[0] or 0),
            "social_sentiment":  round(float(social_df["avg_sentiment"].iloc[0] or 0), 3),
            "bullish_posts":     int(social_df["bullish_count"].iloc[0] or 0),
            "bearish_posts":     int(social_df["bearish_count"].iloc[0] or 0),

            # Price data
            "latest_price":      round(float(price_df.iloc[0]["close_price"]), 2)
                                  if not price_df.empty else None,
            "price_change":      price_change,
            "price_change_pct":  price_change_pct,
            "volume":            int(price_df.iloc[0]["volume"])
                                  if not price_df.empty else None,

            # ML prediction
            "prediction":        pred_df.iloc[0]["predicted_movement"]
                                  if not pred_df.empty else "N/A",
            "pred_confidence":   round(float(pred_df.iloc[0]["confidence"]) * 100, 1)
                                  if not pred_df.empty else None
        }

    conn.close()
    return summary


def get_top_headlines(limit=3):
    """Get top headlines from last 24 hours"""
    conn = get_connection()
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    query = """
        SELECT ticker, title, sentiment, confidence
        FROM news_articles
        WHERE DATE(fetched_at) >= ?
        AND sentiment != 'neutral'
        ORDER BY confidence DESC
        LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(yesterday, limit))
    conn.close()
    return df


# ─────────────────────────────────────────
# LLM REPORT GENERATION
# ─────────────────────────────────────────

def build_prompt(summary, headlines_df):
    """
    Build a detailed prompt for the LLM
    using real data from our DB
    """

    # Format stock data for prompt
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
- News: {data['total_articles']} articles | 
  Sentiment: {sentiment_label} ({data['news_sentiment']:.2f}) | 
  Positive: {data['positive_news']} | 
  Negative: {data['negative_news']}
- Social: {data['total_posts']} posts | 
  Bullish: {data['bullish_posts']} | 
  Bearish: {data['bearish_posts']}
- Price: ${data['latest_price']} 
  {price_arrow} {data['price_change_pct']}% change
- ML Prediction: {data['prediction']} 
  ({data['pred_confidence']}% confidence)
"""

    # Format headlines
    headlines_text = ""
    if not headlines_df.empty:
        headlines_text = "\nTop Headlines:\n"
        for _, row in headlines_df.iterrows():
            headlines_text += (
                f"- [{row['ticker']}] "
                f"[{row['sentiment'].upper()}] "
                f"{row['title']}\n"
            )

    prompt = f"""
You are a professional financial analyst writing a morning 
market intelligence briefing.

Today's Date: {datetime.now().strftime("%B %d, %Y")}

Below is real sentiment and price data collected from 
news sources and social media for 5 major stocks.
Write a professional, concise morning briefing report.

DATA:
{stocks_text}
{headlines_text}

Write the briefing in this exact format:

MARKET INTELLIGENCE BRIEFING
{datetime.now().strftime("%B %d, %Y")}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MARKET OVERVIEW
[2-3 sentences summarizing overall market sentiment today]

STOCK ANALYSIS
[For each stock: 2-3 sentences covering sentiment trend, 
price movement, and what our ML model predicts for tomorrow]

KEY SIGNALS TO WATCH
[3 bullet points of the most important things 
to monitor today]

SENTIMENT SUMMARY TABLE
[Simple text table: Stock | Sentiment | Price Change | Prediction]

Keep it professional, data-driven, and under 400 words.
Do not make specific buy/sell recommendations.
"""
    return prompt


def generate_digest():
    """
    Main function — pulls data, calls LLM,
    saves report to file and DB
    """
    print("\n📝 Generating Daily Market Digest...")
    print("-" * 40)

    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in .env")
        print("   Get free key at console.groq.com")
        return None

    # Step 1 — gather data
    print("📊 Gathering data from database...")
    summary      = get_yesterday_summary()
    headlines_df = get_top_headlines(limit=5)
    print(f"✅ Data gathered for {len(summary)} stocks")

    # Step 2 — build prompt
    prompt = build_prompt(summary, headlines_df)

    # Step 3 — call Groq LLM
    print("🤖 Calling LLM to generate report...")
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional financial analyst. "
                        "Write clear, factual, data-driven market briefings. "
                        "Never make specific investment advice."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=800,
            temperature=0.3
        )

        report = response.choices[0].message.content

    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return None

    # Step 4 — save to file
    date_str    = datetime.now().strftime("%Y-%m-%d")
    output_path = f"{OUTPUT_DIR}/digest_{date_str}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ Report saved to: {output_path}")

    # Step 5 — save to database
    save_digest(date_str, report)

    # Step 6 — print report
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    return report


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    report = generate_digest()

    if report:
        print("\n✅ Digest generation complete!")
        print(f"📁 Saved in: {OUTPUT_DIR}/")
    else:
        print("\n❌ Digest generation failed")
        print("   Check your GROQ_API_KEY in .env")