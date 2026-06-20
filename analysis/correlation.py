import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sqlite3
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import get_connection, get_feature_table

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

STOCKS = ["AAPL", "TSLA", "MSFT", "NVDA", "AMZN"]

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────

def load_merged_data(ticker):
    """
    Load and merge sentiment + price data
    for a single ticker
    """
    conn = get_connection()

    # Daily average news sentiment
    news_query = """
        SELECT 
            DATE(published_at)      as date,
            AVG(sentiment_score)    as news_sentiment,
            COUNT(*)                as news_count
        FROM news_articles
        WHERE ticker = ?
        AND sentiment IS NOT NULL
        GROUP BY DATE(published_at)
    """

    # Daily average social sentiment
    social_query = """
        SELECT 
            DATE(created_at)        as date,
            AVG(sentiment_score)    as social_sentiment,
            COUNT(*)                as social_count
        FROM social_posts
        WHERE ticker = ?
        AND sentiment IS NOT NULL
        GROUP BY DATE(created_at)
    """

    # Daily stock prices
    price_query = """
        SELECT 
            date,
            close_price,
            volume
        FROM stock_prices
        WHERE ticker = ?
        ORDER BY date
    """

    news_df   = pd.read_sql(news_query,   conn, params=(ticker,))
    social_df = pd.read_sql(social_query, conn, params=(ticker,))
    price_df  = pd.read_sql(price_query,  conn, params=(ticker,))
    conn.close()

    if price_df.empty:
        print(f"⚠️  No price data found for {ticker}")
        return pd.DataFrame()

    # Merge all on date
    df = price_df.copy()
    df = df.merge(news_df,   on="date", how="left")
    df = df.merge(social_df, on="date", how="left")

    # Fill missing sentiment with 0 (neutral)
    df["news_sentiment"]   = df["news_sentiment"].fillna(0)
    df["social_sentiment"] = df["social_sentiment"].fillna(0)
    df["news_count"]       = df["news_count"].fillna(0)
    df["social_count"]     = df["social_count"].fillna(0)

    # Calculate daily price change %
    df = df.sort_values("date")
    df["price_change_pct"] = df["close_price"].pct_change() * 100

    # Calculate previous day return
    df["prev_day_return"] = df["price_change_pct"].shift(1)

    # Direction — Up(1) or Down(0)
    df["direction"] = (df["price_change_pct"] > 0).astype(int)

    # Lagged sentiment — yesterday's sentiment
    df["news_sentiment_lag1"]   = df["news_sentiment"].shift(1)
    df["social_sentiment_lag1"] = df["social_sentiment"].shift(1)

    df = df.dropna()
    df = df.reset_index(drop=True)

    return df


# ─────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────

def calculate_correlation(ticker):
    """
    Calculate correlation between sentiment
    and price movement for one ticker
    """
    df = load_merged_data(ticker)

    if df.empty or len(df) < 5:
        print(f"⚠️  Not enough data for {ticker}")
        return None

    results = {"ticker": ticker, "data_points": len(df)}

    # 1 — Same day correlation
    # Does today's sentiment match today's price move?
    corr1, p1 = stats.pearsonr(
        df["news_sentiment"],
        df["price_change_pct"]
    )
    results["same_day_news_corr"]  = round(corr1, 3)
    results["same_day_p_value"]    = round(p1, 4)

    # 2 — Lagged correlation
    # Does yesterday's sentiment predict today's price?
    corr2, p2 = stats.pearsonr(
        df["news_sentiment_lag1"],
        df["price_change_pct"]
    )
    results["lagged_news_corr"]    = round(corr2, 3)
    results["lagged_p_value"]      = round(p2, 4)

    # 3 — Social sentiment correlation
    corr3, p3 = stats.pearsonr(
        df["social_sentiment"],
        df["price_change_pct"]
    )
    results["social_sentiment_corr"] = round(corr3, 3)

    # 4 — Combined sentiment correlation
    df["combined_sentiment"] = (
        df["news_sentiment"] * 0.6 +
        df["social_sentiment"] * 0.4
    )
    corr4, p4 = stats.pearsonr(
        df["combined_sentiment"],
        df["price_change_pct"]
    )
    results["combined_corr"] = round(corr4, 3)

    # 5 — Interpretation
    results["interpretation"] = interpret(corr2)

    # 6 — Signal strength
    results["signal_strength"] = signal_strength(corr2, p2)

    return results


def interpret(corr):
    """Human readable interpretation of correlation"""
    if corr > 0.6:
        return "Very strong — positive news strongly predicts price rise"
    elif corr > 0.4:
        return "Strong positive relationship"
    elif corr > 0.2:
        return "Moderate positive relationship"
    elif corr > 0:
        return "Weak positive relationship"
    elif corr > -0.2:
        return "Weak negative relationship"
    elif corr > -0.4:
        return "Moderate negative relationship"
    elif corr > -0.6:
        return "Strong negative relationship"
    else:
        return "Very strong negative relationship"


def signal_strength(corr, p_value):
    """Rate the signal strength"""
    abs_corr = abs(corr)
    if p_value > 0.05:
        return "⚪ Not statistically significant"
    elif abs_corr > 0.5:
        return "🔴 Very strong signal"
    elif abs_corr > 0.3:
        return "🟠 Strong signal"
    elif abs_corr > 0.15:
        return "🟡 Moderate signal"
    else:
        return "🟢 Weak signal"


def run_all_correlations():
    """Run correlation analysis for all tickers"""
    print("\n📊 Running Correlation Analysis...")
    print("=" * 60)

    all_results = []

    for ticker in STOCKS:
        print(f"\nAnalyzing {ticker}...")
        result = calculate_correlation(ticker)
        if result:
            all_results.append(result)
            print(f"  Same-day news corr:  {result['same_day_news_corr']}")
            print(f"  Lagged news corr:    {result['lagged_news_corr']}")
            print(f"  Social corr:         {result['social_sentiment_corr']}")
            print(f"  Combined corr:       {result['combined_corr']}")
            print(f"  Signal:              {result['signal_strength']}")
            print(f"  Interpretation:      {result['interpretation']}")

    if not all_results:
        print("❌ No correlation results generated")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Save results
    results_df.to_csv(
        f"{OUTPUT_DIR}/correlation_results.csv",
        index=False
    )
    print(f"\n✅ Results saved to {OUTPUT_DIR}/correlation_results.csv")

    return results_df


# ─────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────

def plot_sentiment_vs_price(ticker):
    """
    Plot sentiment vs price movement
    for one ticker — saves as PNG
    """
    df = load_merged_data(ticker)

    if df.empty:
        print(f"⚠️  No data to plot for {ticker}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        f"{ticker} — Sentiment vs Price Analysis",
        fontsize=14, fontweight="bold"
    )

    # Plot 1 — Price change over time
    axes[0].bar(
        df["date"],
        df["price_change_pct"],
        color=["green" if x > 0 else "red"
               for x in df["price_change_pct"]],
        alpha=0.7
    )
    axes[0].set_title("Daily Price Change %")
    axes[0].set_ylabel("Price Change %")
    axes[0].axhline(y=0, color="black", linewidth=0.5)
    axes[0].tick_params(axis="x", rotation=45)

    # Plot 2 — News sentiment over time
    axes[1].bar(
        df["date"],
        df["news_sentiment"],
        color=["green" if x > 0 else "red" if x < 0 else "gray"
               for x in df["news_sentiment"]],
        alpha=0.7
    )
    axes[1].set_title("Daily News Sentiment Score")
    axes[1].set_ylabel("Sentiment Score")
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].tick_params(axis="x", rotation=45)

    # Plot 3 — Scatter — sentiment vs price
    axes[2].scatter(
        df["news_sentiment_lag1"],
        df["price_change_pct"],
        alpha=0.6,
        color="royalblue",
        edgecolors="white",
        s=60
    )

    # Add trend line
    if len(df) > 2:
        z = np.polyfit(
            df["news_sentiment_lag1"],
            df["price_change_pct"], 1
        )
        p = np.poly1d(z)
        x_line = np.linspace(
            df["news_sentiment_lag1"].min(),
            df["news_sentiment_lag1"].max(), 100
        )
        axes[2].plot(x_line, p(x_line), "r--", alpha=0.8)

    axes[2].set_title(
        "Lagged Sentiment vs Next Day Price Change"
    )
    axes[2].set_xlabel("Yesterday Sentiment Score")
    axes[2].set_ylabel("Today Price Change %")
    axes[2].axhline(y=0, color="black", linewidth=0.3)
    axes[2].axvline(x=0, color="black", linewidth=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = f"{OUTPUT_DIR}/{ticker}_correlation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot saved: {plot_path}")


def plot_correlation_heatmap(results_df):
    """
    Heatmap comparing correlations
    across all stocks
    """
    if results_df.empty:
        return

    heatmap_data = results_df.set_index("ticker")[[
        "same_day_news_corr",
        "lagged_news_corr",
        "social_sentiment_corr",
        "combined_corr"
    ]]

    heatmap_data.columns = [
        "Same Day News",
        "Lagged News",
        "Social",
        "Combined"
    ]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"}
    )
    plt.title(
        "Sentiment vs Price Correlation Heatmap\n"
        "(All Stocks)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    heatmap_path = f"{OUTPUT_DIR}/correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Heatmap saved: {heatmap_path}")


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 — Run all correlations
    results_df = run_all_correlations()

    # Step 2 — Generate plots for each stock
    print("\n📈 Generating charts...")
    print("-" * 40)
    for ticker in STOCKS:
        plot_sentiment_vs_price(ticker)

    # Step 3 — Generate heatmap
    plot_correlation_heatmap(results_df)

    print("\n✅ Correlation analysis complete!")
    print(f"📁 All outputs saved in: {OUTPUT_DIR}/")
    print("\nFiles generated:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  → {f}")