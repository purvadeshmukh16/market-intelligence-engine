import yfinance as yf
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import save_stock_prices

STOCKS = {
    "AAPL":  "Apple",
    "TSLA":  "Tesla",
    "MSFT":  "Microsoft",
    "NVDA":  "NVIDIA",
    "AMZN":  "Amazon",
    "GOOGL": "Google",      # ← add this
    "META":  "Meta",        # ← add this
    "RELIANCE.NS": "Reliance",  # ← Indian stock
    "TCS.NS": "TCS"         
}

def fetch_stock_data(ticker, period="1mo"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            print(f"⚠️  No data returned for {ticker}")
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={
            "Date":   "date",
            "Open":   "open_price",
            "High":   "high_price",
            "Low":    "low_price",
            "Close":  "close_price",
            "Volume": "volume"
        })

        df = df[["date", "open_price", "high_price",
                 "low_price", "close_price", "volume"]]

        df["ticker"]     = ticker
        df["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ✅ FIX — remove timezone before formatting
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

        print(f"✅ Fetched {len(df)} days of data for {ticker}")
        return df

    except Exception as e:
        print(f"❌ Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def fetch_all_stocks(period="1mo"):
    all_data = []

    print("\n📈 Fetching stock prices...")
    print("-" * 40)

    for ticker, company in STOCKS.items():
        print(f"Fetching {company} ({ticker})...")
        df = fetch_stock_data(ticker, period)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("❌ No stock data fetched")
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    print("-" * 40)
    print(f"✅ Total records fetched: {len(final_df)}")
    return final_df


def get_latest_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2d")

        if df.empty:
            return None

        latest = df.iloc[-1]
        prev   = df.iloc[-2] if len(df) > 1 else df.iloc[-1]

        return {
            "ticker":        ticker,
            "current_price": round(latest["Close"], 2),
            "prev_close":    round(prev["Close"], 2),
            "price_change":  round(latest["Close"] - prev["Close"], 2),
            "pct_change":    round(
                              ((latest["Close"] - prev["Close"])
                                / prev["Close"]) * 100, 2
                             ),
            "volume":        int(latest["Volume"]),
            "fetched_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"❌ Failed to get latest price for {ticker}: {e}")
        return None


def get_all_latest_prices():
    results = []
    for ticker in STOCKS:
        data = get_latest_price(ticker)
        if data:
            results.append(data)
    return results


if __name__ == "__main__":

    # Test 1 — fetch all stocks
    df = fetch_all_stocks(period="1mo")

    if not df.empty:
        print("\n📊 Sample data:")
        print(df.head(10).to_string(index=False))

        # Test 2 — save to database
        print("\n💾 Saving to database...")
        save_stock_prices(df)

        # Test 3 — save to CSV backup
        csv_path = "data/raw/stock_prices.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV backup saved to {csv_path}")

    # Test 4 — latest prices
    print("\n🔴 Live prices:")
    print("-" * 40)
    prices = get_all_latest_prices()
    for p in prices:
        arrow = "📈" if p["pct_change"] > 0 else "📉"
        print(f"{arrow}  {p['ticker']}: "
              f"${p['current_price']} "
              f"({p['pct_change']:+.2f}%)")