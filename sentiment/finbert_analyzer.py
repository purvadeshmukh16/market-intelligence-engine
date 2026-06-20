from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import sqlite3
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import get_connection

# ─────────────────────────────────────────
# LOAD FINBERT MODEL
# ─────────────────────────────────────────

print("⏳ Loading FinBERT model (first time takes 2-3 mins to download)...")

MODEL_NAME = "ProsusAI/finbert"

try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("✅ FinBERT model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load FinBERT: {e}")
    sys.exit(1)

# Labels that FinBERT outputs
LABELS = ["positive", "negative", "neutral"]

# ─────────────────────────────────────────
# CORE SENTIMENT FUNCTION
# ─────────────────────────────────────────

def get_sentiment(text):
    """
    Run FinBERT on a single piece of text
    Returns: label, score, confidence

    label      → positive / negative / neutral
    score      → +1 / -1 / 0  (numeric for math)
    confidence → 0.0 to 1.0   (how sure the model is)
    """

    # Handle empty or too short text
    if not text or len(str(text).strip()) < 5:
        return "neutral", 0, 0.0

    try:
        # Tokenize text
        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Run model — no gradient needed (we are not training)
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert raw outputs to probabilities
        probs = torch.softmax(outputs.logits, dim=1).squeeze()

        # Get label and confidence
        label_index = probs.argmax().item()
        label       = LABELS[label_index]
        confidence  = round(probs[label_index].item(), 3)

        # Convert label to numeric score
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        score     = score_map[label]

        return label, score, confidence

    except Exception as e:
        print(f"⚠️  Error on text: {str(text)[:50]}... → {e}")
        return "neutral", 0, 0.0


def score_texts(texts):
    """
    Run FinBERT on a list of texts
    Returns list of dicts with sentiment results
    """
    results = []
    for text in texts:
        label, score, confidence = get_sentiment(text)
        results.append({
            "sentiment":       label,
            "sentiment_score": score,
            "confidence":      confidence
        })
    return results


# ─────────────────────────────────────────
# DATABASE UPDATE FUNCTIONS
# ─────────────────────────────────────────

def score_news_in_db():
    """
    Read all unscored news from DB
    Run FinBERT on each title
    Save scores back to DB
    """
    conn = get_connection()

    # Get unscored news articles
    query = """
        SELECT id, ticker, title 
        FROM news_articles 
        WHERE sentiment IS NULL
    """
    df = pd.read_sql(query, conn)

    if df.empty:
        print("⚠️  No unscored news articles found in DB")
        conn.close()
        return

    print(f"\n📰 Scoring {len(df)} news articles with FinBERT...")
    print("-" * 40)

    cursor = conn.cursor()
    scored = 0

    for _, row in df.iterrows():
        label, score, confidence = get_sentiment(row["title"])

        cursor.execute("""
            UPDATE news_articles
            SET sentiment       = ?,
                sentiment_score = ?,
                confidence      = ?
            WHERE id = ?
        """, (label, score, confidence, row["id"]))

        scored += 1

        # Show progress every 10 articles
        if scored % 10 == 0:
            print(f"  Scored {scored}/{len(df)} articles...")

    conn.commit()
    conn.close()

    print(f"✅ Scored all {scored} news articles")


def score_social_in_db():
    """
    Read all unscored social posts from DB
    Run FinBERT on each post
    Save scores back to DB
    """
    conn = get_connection()

    # Get unscored social posts
    query = """
        SELECT id, ticker, text 
        FROM social_posts 
        WHERE sentiment IS NULL
    """
    df = pd.read_sql(query, conn)

    if df.empty:
        print("⚠️  No unscored social posts found in DB")
        conn.close()
        return

    print(f"\n💬 Scoring {len(df)} social posts with FinBERT...")
    print("-" * 40)

    cursor = conn.cursor()
    scored = 0

    for _, row in df.iterrows():
        label, score, confidence = get_sentiment(row["text"])

        cursor.execute("""
            UPDATE social_posts
            SET sentiment       = ?,
                sentiment_score = ?,
                confidence      = ?
            WHERE id = ?
        """, (label, score, confidence, row["id"]))

        scored += 1

        if scored % 10 == 0:
            print(f"  Scored {scored}/{len(df)} posts...")

    conn.commit()
    conn.close()

    print(f"✅ Scored all {scored} social posts")


def score_all():
    """Score all unscored data in DB — news + social"""
    print("\n🤖 Running FinBERT sentiment analysis...")
    score_news_in_db()
    score_social_in_db()
    print("\n✅ All data scored successfully")


# ─────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────

def get_sentiment_summary():
    """Print sentiment breakdown by ticker after scoring"""
    conn = get_connection()

    print("\n📊 Sentiment Summary by Ticker:")
    print("=" * 50)

    query = """
        SELECT 
            ticker,
            sentiment,
            COUNT(*) as count,
            ROUND(AVG(confidence), 2) as avg_confidence
        FROM news_articles
        WHERE sentiment IS NOT NULL
        GROUP BY ticker, sentiment
        ORDER BY ticker, sentiment
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("No scored data found")
        return

    for ticker in df["ticker"].unique():
        tdf   = df[df["ticker"] == ticker]
        total = tdf["count"].sum()
        print(f"\n{ticker} ({total} articles):")
        for _, row in tdf.iterrows():
            emoji = "🟢" if row["sentiment"] == "positive" \
                    else "🔴" if row["sentiment"] == "negative" \
                    else "🟡"
            pct = round((row["count"] / total) * 100, 1)
            print(f"  {emoji} {row['sentiment']:10} "
                  f"{row['count']:3} articles "
                  f"({pct}%) "
                  f"avg confidence: {row['avg_confidence']}")


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Test 1 — test on sample sentences
    print("\n🧪 Testing FinBERT on sample texts:")
    print("-" * 40)

    test_texts = [
        "Apple hits record revenue driven by strong iPhone sales",
        "Tesla recalls 50000 vehicles over safety concerns",
        "Microsoft announces quarterly earnings report",
        "NVIDIA stock surges after beating analyst expectations",
        "Amazon faces antitrust investigation from regulators"
    ]

    for text in test_texts:
        label, score, confidence = get_sentiment(text)
        emoji = "🟢" if label == "positive" \
                else "🔴" if label == "negative" \
                else "🟡"
        print(f"{emoji} [{label:8}] [{confidence:.0%}] {text[:60]}...")

    # Test 2 — score everything in DB
    print("\n")
    score_all()

    # Test 3 — show summary
    get_sentiment_summary()