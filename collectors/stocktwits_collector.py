import requests
import pandas as pd
from datetime import datetime
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import save_social_posts

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

STOCKS = {
   "AAPL":  "Apple",
    "TSLA":  "Tesla",
    "MSFT":  "Microsoft",
    "NVDA":  "NVIDIA",
    "AMZN":  "Amazon",
    "GOOGL": "Google",
    "META":  "Meta"        
}

# ─────────────────────────────────────────
# FETCH FUNCTIONS
# ─────────────────────────────────────────

def fetch_stocktwits(ticker):
    """
    Fetch latest posts for a ticker from StockTwits
    No API key needed — completely free
    """
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data     = response.json()
            messages = data.get("messages", [])

            results = []
            for msg in messages:

                # StockTwits already provides bullish/bearish labels
                sentiment_data = msg.get("entities", {}) \
                                    .get("sentiment", None)
                user_sentiment = None
                if sentiment_data:
                    user_sentiment = sentiment_data.get("basic", None)

                results.append({
                    "ticker":          ticker,
                    "text":            msg["body"],
                    "platform":        "stocktwits",
                    "likes":           msg.get("likes", {}).get("total", 0),
                    "user_sentiment":  user_sentiment,  # Bullish/Bearish/None
                    "created_at":      msg["created_at"],
                    "fetched_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    # FinBERT scores — filled later
                    "sentiment":       None,
                    "sentiment_score": None,
                    "confidence":      None
                })

            return results

        elif response.status_code == 429:
            print(f"⚠️  Rate limited for {ticker} — waiting 5 seconds...")
            time.sleep(5)
            return []

        elif response.status_code == 404:
            print(f"⚠️  {ticker} not found on StockTwits")
            return []

        else:
            print(f"⚠️  StockTwits error {response.status_code} for {ticker}")
            return []

    except requests.exceptions.ConnectionError:
        print(f"❌ No internet connection")
        return []
    except requests.exceptions.Timeout:
        print(f"⚠️  Request timed out for {ticker}")
        return []
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return []


def fetch_all_stocktwits():
    """Fetch StockTwits posts for all 5 stocks"""

    print("\n💬 Fetching StockTwits posts...")
    print("-" * 40)

    all_posts = []

    for ticker, company in STOCKS.items():
        print(f"Fetching {company} ({ticker})...")

        posts = fetch_stocktwits(ticker)
        all_posts.extend(posts)
        print(f"  ✅ {len(posts)} posts fetched")

        # Small delay to avoid rate limiting
        time.sleep(1)

    if not all_posts:
        print("❌ No posts fetched")
        return pd.DataFrame()

    df = pd.DataFrame(all_posts)
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)

    print("-" * 40)
    print(f"✅ Total unique posts: {len(df)}")

    print("\nPosts per ticker:")
    print(df["ticker"].value_counts().to_string())

    # Show sentiment distribution from StockTwits labels
    print("\nUser sentiment labels from StockTwits:")
    print(df["user_sentiment"].value_counts().to_string())

    return df


def get_sentiment_summary(ticker):
    """
    Quick summary of StockTwits sentiment for one ticker
    Uses the built-in Bullish/Bearish labels
    """
    posts = fetch_stocktwits(ticker)

    if not posts:
        return None

    df     = pd.DataFrame(posts)
    total  = len(df)
    bull   = len(df[df["user_sentiment"] == "Bullish"])
    bear   = len(df[df["user_sentiment"] == "Bearish"])
    neutral = total - bull - bear

    return {
        "ticker":       ticker,
        "total_posts":  total,
        "bullish":      bull,
        "bearish":      bear,
        "neutral":      neutral,
        "bull_pct":     round((bull / total) * 100, 1) if total > 0 else 0,
        "bear_pct":     round((bear / total) * 100, 1) if total > 0 else 0,
        "overall_mood": "Bullish" if bull > bear else
                        "Bearish" if bear > bull else "Neutral"
    }


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Test 1 — fetch all posts
    df = fetch_all_stocktwits()

    if not df.empty:

        # Show sample posts
        print("\n📋 Sample posts:")
        print("-" * 40)
        for _, row in df.head(5).iterrows():
            mood = row["user_sentiment"] or "No label"
            print(f"[{row['ticker']}] [{mood}] {row['text'][:80]}...")

        # Save to database
        print("\n💾 Saving to database...")
        save_social_posts(df)

        # Save CSV backup
        csv_path = "data/raw/stocktwits_posts.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV backup saved to {csv_path}")

    # Test 2 — quick sentiment summary per stock
    print("\n📊 Sentiment Summary (from StockTwits labels):")
    print("-" * 40)
    for ticker in STOCKS:
        summary = get_sentiment_summary(ticker)
        if summary:
            print(f"{ticker}: {summary['overall_mood']} "
                  f"| 🟢 {summary['bull_pct']}% Bullish "
                  f"| 🔴 {summary['bear_pct']}% Bearish "
                  f"| {summary['total_posts']} posts")
        time.sleep(1)