import requests
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import save_news

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Keywords mapped to tickers
STOCK_KEYWORDS = {
     "AAPL":         ["Apple stock", "Apple Inc earnings", "AAPL"],
    "TSLA":         ["Tesla stock", "Tesla earnings", "TSLA"],
    "MSFT":         ["Microsoft stock", "Microsoft earnings", "MSFT"],
    "NVDA":         ["NVIDIA stock", "NVIDIA earnings", "NVDA"],
    "AMZN":         ["Amazon stock", "Amazon earnings", "AMZN"],
    "GOOGL":        ["Google stock", "Google earnings", "GOOGL"],
    "META":         ["Meta stock", "Meta earnings", "META"],
    "RELIANCE.NS":  ["Reliance Industries stock", "Reliance earnings"],
    "TCS.NS":       ["TCS stock", "Tata Consultancy earnings", "TCS"]
}

# ─────────────────────────────────────────
# FETCH FUNCTIONS
# ─────────────────────────────────────────

def fetch_news_for_ticker(ticker, keywords):
    """Fetch news articles for one ticker using multiple keywords"""

    # Get news from last 7 days
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    all_articles = []

    for keyword in keywords:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q":          keyword,
                "apiKey":     NEWS_API_KEY,
                "language":   "en",
                "sortBy":     "publishedAt",
                "pageSize":   10,
                "from":       from_date
            }

            response = requests.get(url, params=params, timeout=10)

            # Check for API errors
            if response.status_code == 401:
                print("❌ Invalid NewsAPI key — check your .env file")
                return []
            if response.status_code == 429:
                print("⚠️  NewsAPI rate limit hit — wait and try again")
                return []
            if response.status_code != 200:
                print(f"⚠️  NewsAPI error {response.status_code} for {keyword}")
                continue

            articles = response.json().get("articles", [])

            for article in articles:
                # Skip articles with missing title
                if not article.get("title"):
                    continue
                if article["title"] == "[Removed]":
                    continue

                all_articles.append({
                    "ticker":       ticker,
                    "title":        article["title"],
                    "source":       article["source"]["name"],
                    "published_at": article["publishedAt"],
                    "fetched_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    # Sentiment will be filled by FinBERT later
                    "sentiment":       None,
                    "sentiment_score": None,
                    "confidence":      None
                })

        except requests.exceptions.ConnectionError:
            print(f"❌ No internet connection")
            return []
        except requests.exceptions.Timeout:
            print(f"⚠️  Request timed out for {keyword}")
            continue
        except Exception as e:
            print(f"❌ Error fetching {keyword}: {e}")
            continue

    # Remove duplicate titles
    seen = set()
    unique_articles = []
    for article in all_articles:
        if article["title"] not in seen:
            seen.add(article["title"])
            unique_articles.append(article)

    return unique_articles


def fetch_google_rss(ticker, company_name):
    """
    Backup source — Google News RSS
    No API key needed at all
    """
    try:
        import feedparser
        url = (f"https://news.google.com/rss/search?"
               f"q={company_name}+stock&hl=en-US&gl=US&ceid=US:en")
        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries[:10]:
            articles.append({
                "ticker":          ticker,
                "title":           entry.title,
                "source":          "Google News",
                "published_at":    entry.get("published", 
                                   datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")),
                "fetched_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment":       None,
                "sentiment_score": None,
                "confidence":      None
            })
        return articles

    except Exception as e:
        print(f"⚠️  Google RSS failed for {ticker}: {e}")
        return []


def fetch_all_news():
    """Fetch news for all 5 stocks from NewsAPI + Google RSS"""

    print("\n📰 Fetching news articles...")
    print("-" * 40)

    if not NEWS_API_KEY:
        print("⚠️  NEWS_API_KEY not found in .env")
        print("    Falling back to Google RSS only...")

    all_articles = []

    COMPANIES = {
    "AAPL":         "Apple",
    "TSLA":         "Tesla",
    "MSFT":         "Microsoft",
    "NVDA":         "NVIDIA",
    "AMZN":         "Amazon",
    "GOOGL":        "Google",
    "META":         "Meta",
    "RELIANCE.NS":  "Reliance Industries",
    "TCS.NS":       "TCS"
    }

    for ticker, keywords in STOCK_KEYWORDS.items():
        print(f"\nFetching news for {ticker}...")

        # Source 1 — NewsAPI
        if NEWS_API_KEY:
            newsapi_articles = fetch_news_for_ticker(ticker, keywords)
            all_articles.extend(newsapi_articles)
            print(f"  NewsAPI:    {len(newsapi_articles)} articles")

        # Source 2 — Google RSS (always runs as backup)
        rss_articles = fetch_google_rss(ticker, COMPANIES[ticker])
        all_articles.extend(rss_articles)
        print(f"  Google RSS: {len(rss_articles)} articles")

    if not all_articles:
        print("\n❌ No articles fetched")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)

    # Final dedup across all tickers
    df = df.drop_duplicates(subset=["title"])
    df = df.reset_index(drop=True)

    print("\n" + "-" * 40)
    print(f"✅ Total unique articles: {len(df)}")
    print("\nArticles per ticker:")
    print(df["ticker"].value_counts().to_string())

    return df


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Check API key
    if not NEWS_API_KEY:
        print("⚠️  WARNING: NEWS_API_KEY not set in .env")
        print("   Google RSS will still work as backup\n")
    else:
        print(f"✅ NewsAPI key loaded: {NEWS_API_KEY[:8]}...")

    # Fetch all news
    df = fetch_all_news()

    if not df.empty:

        # Show sample
        print("\n📋 Sample headlines:")
        print("-" * 40)
         
        for _, row in df.head(5).iterrows():
            print(f"[{row['ticker']}] {row['title'][:80]}...")

        # Save to database
        print("\n💾 Saving to database...")
        save_news(df)

        # Save CSV backup
        csv_path = "data/raw/news_articles.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV backup saved to {csv_path}")
