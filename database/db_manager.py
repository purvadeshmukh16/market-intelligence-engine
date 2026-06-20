import sqlite3
import pandas as pd
from datetime import datetime
import os

# Database will be created inside data folder
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'market_sentiment.db')

def get_connection():
    """Create and return database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    """Create all tables if they don't exist"""
    conn = get_connection()
    cursor = conn.cursor()

    # Table 1 - News articles
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            published_at TEXT,
            sentiment TEXT,
            sentiment_score REAL,
            confidence REAL,
            fetched_at TEXT
        )
    """)

    # Table 2 - StockTwits posts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS social_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            text TEXT NOT NULL,
            platform TEXT DEFAULT 'stocktwits',
            likes INTEGER DEFAULT 0,
            user_sentiment TEXT,
            sentiment TEXT,
            sentiment_score REAL,
            confidence REAL,
            created_at TEXT,
            fetched_at TEXT
        )
    """)

    # Table 3 - Stock prices
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            open_price REAL,
            close_price REAL,
            high_price REAL,
            low_price REAL,
            volume INTEGER,
            date TEXT,
            fetched_at TEXT
        )
    """)

    # Table 4 - ML predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            predicted_movement TEXT,
            confidence REAL,
            model_used TEXT,
            news_sentiment REAL,
            social_sentiment REAL,
            volume INTEGER,
            prev_day_return REAL,
            predicted_at TEXT
        )
    """)

    # Table 5 - Daily digests
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_digests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT,
            content TEXT,
            generated_at TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("✅ All tables created successfully")


# ─────────────────────────────────────────
# SAVE FUNCTIONS
# ─────────────────────────────────────────

def save_news(df: pd.DataFrame):
    """Save news articles to database"""
    if df.empty:
        print("⚠️ No news data to save")
        return
    conn = get_connection()
    df.to_sql("news_articles", conn, if_exists="append", index=False)
    conn.close()
    print(f"✅ Saved {len(df)} news articles to DB")


def save_social_posts(df: pd.DataFrame):
    """Save StockTwits posts to database"""
    if df.empty:
        print("⚠️ No social data to save")
        return
    conn = get_connection()
    df.to_sql("social_posts", conn, if_exists="append", index=False)
    conn.close()
    print(f"✅ Saved {len(df)} social posts to DB")


def save_stock_prices(df: pd.DataFrame):
    """Save stock prices to database"""
    if df.empty:
        print("⚠️ No price data to save")
        return
    conn = get_connection()
    df.to_sql("stock_prices", conn, if_exists="append", index=False)
    conn.close()
    print(f"✅ Saved {len(df)} stock price records to DB")


def save_prediction(ticker, movement, confidence, model_used, features: dict):
    """Save a single prediction to database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions 
        (ticker, predicted_movement, confidence, model_used,
         news_sentiment, social_sentiment, volume, 
         prev_day_return, predicted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker,
        movement,
        confidence,
        model_used,
        features.get("news_sentiment", 0),
        features.get("social_sentiment", 0),
        features.get("volume", 0),
        features.get("prev_day_return", 0),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()
    print(f"✅ Saved prediction for {ticker}: {movement} ({confidence:.0%} confidence)")


def save_digest(report_date, content):
    """Save daily digest report"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO daily_digests (report_date, content, generated_at)
        VALUES (?, ?, ?)
    """, (report_date, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    print(f"✅ Saved digest for {report_date}")


# ─────────────────────────────────────────
# FETCH FUNCTIONS
# ─────────────────────────────────────────

def get_news_sentiment_by_date(ticker, days=14):
    """Get average news sentiment per day for a ticker"""
    conn = get_connection()
    query = """
        SELECT 
            DATE(published_at) as date,
            AVG(sentiment_score) as avg_sentiment,
            COUNT(*) as article_count
        FROM news_articles
        WHERE ticker = ?
        GROUP BY DATE(published_at)
        ORDER BY date DESC
        LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(ticker, days))
    conn.close()
    return df


def get_social_sentiment_by_date(ticker, days=14):
    """Get average social sentiment per day for a ticker"""
    conn = get_connection()
    query = """
        SELECT 
            DATE(created_at) as date,
            AVG(sentiment_score) as avg_sentiment,
            COUNT(*) as post_count
        FROM social_posts
        WHERE ticker = ?
        GROUP BY DATE(created_at)
        ORDER BY date DESC
        LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(ticker, days))
    conn.close()
    return df


def get_stock_prices_by_date(ticker, days=14):
    """Get stock prices for a ticker"""
    conn = get_connection()
    query = """
        SELECT date, open_price, close_price, 
               high_price, low_price, volume
        FROM stock_prices
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(ticker, days))
    conn.close()
    return df


def get_latest_prediction(ticker):
    """Get most recent prediction for a ticker"""
    conn = get_connection()
    query = """
        SELECT * FROM predictions
        WHERE ticker = ?
        ORDER BY predicted_at DESC
        LIMIT 1
    """
    df = pd.read_sql(query, conn, params=(ticker,))
    conn.close()
    return df


def get_all_tickers_summary():
    """Get today's sentiment summary for all tickers"""
    conn = get_connection()
    query = """
        SELECT 
            ticker,
            AVG(sentiment_score) as avg_sentiment,
            COUNT(*) as total_articles
        FROM news_articles
        WHERE DATE(fetched_at) = DATE('now')
        GROUP BY ticker
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_feature_table():
    """
    Build ML feature table by joining 
    sentiment + prices for prediction model
    """
    conn = get_connection()
    query = """
        SELECT 
            sp.ticker,
            sp.date,
            sp.close_price,
            sp.volume,
            COALESCE(ns.avg_sentiment, 0) as news_sentiment,
            COALESCE(ss.avg_sentiment, 0) as social_sentiment
        FROM stock_prices sp
        LEFT JOIN (
            SELECT ticker, DATE(published_at) as date,
                   AVG(sentiment_score) as avg_sentiment
            FROM news_articles
            GROUP BY ticker, DATE(published_at)
        ) ns ON sp.ticker = ns.ticker AND sp.date = ns.date
        LEFT JOIN (
            SELECT ticker, DATE(created_at) as date,
                   AVG(sentiment_score) as avg_sentiment
            FROM social_posts
            GROUP BY ticker, DATE(created_at)
        ) ss ON sp.ticker = ss.ticker AND sp.date = ss.date
        ORDER BY sp.ticker, sp.date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Creating database tables...")
    create_tables()
    print("\nDatabase location:", os.path.abspath(DB_PATH))
    print("\nAll functions loaded successfully ✅")