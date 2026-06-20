import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import sys
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.correlation import load_merged_data

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

STOCKS     = ["AAPL", "TSLA", "MSFT", "NVDA", "AMZN"]
OUTPUT_DIR = "data/processed"
MODEL_DIR  = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# Features used for prediction
FEATURE_COLS = [
    "news_sentiment",
    "social_sentiment",
    "news_sentiment_lag1",
    "social_sentiment_lag1",
    "volume",
    "prev_day_return",
    "news_count",
    "social_count"
]

# ─────────────────────────────────────────
# BUILD FEATURE TABLE
# ─────────────────────────────────────────

def build_feature_table():
    """
    Combine data from all 5 stocks
    into one big feature table for training
    """
    print("\n🔧 Building feature table...")
    all_dfs = []

    for ticker in STOCKS:
        df = load_merged_data(ticker)
        if not df.empty:
            df["ticker"] = ticker
            all_dfs.append(df)
            print(f"  ✅ {ticker}: {len(df)} rows")

    if not all_dfs:
        print("❌ No data found — run collectors first")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Keep only rows where all features exist
    combined = combined.dropna(subset=FEATURE_COLS + ["direction"])

    print(f"\n✅ Total training samples: {len(combined)}")
    print(f"   Up days:   {combined['direction'].sum()}")
    print(f"   Down days: {(combined['direction'] == 0).sum()}")

    return combined


# ─────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────

def train_models(df):
    """
    Train Random Forest, XGBoost,
    and Logistic Regression
    Compare all 3 and save the best one
    """

    if df.empty:
        print("❌ No data to train on")
        return None, None, None

    print("\n🤖 Training ML Models...")
    print("=" * 60)

    # Prepare features and target
    X = df[FEATURE_COLS]
    y = df["direction"]

    # Train test split — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # Scale features for Logistic Regression
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Define 3 models ──
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }

    results     = {}
    best_model  = None
    best_name   = None
    best_acc    = 0

    for name, model in models.items():
        print(f"\n📌 Training {name}...")

        # Use scaled data for Logistic Regression
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=3, scoring="accuracy"
            )
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=3, scoring="accuracy"
            )

        acc = accuracy_score(y_test, preds)

        results[name] = {
            "model":       model,
            "accuracy":    round(acc * 100, 2),
            "cv_mean":     round(cv_scores.mean() * 100, 2),
            "cv_std":      round(cv_scores.std() * 100, 2),
            "predictions": preds
        }

        print(f"  Accuracy:      {acc * 100:.2f}%")
        print(f"  CV Score:      {cv_scores.mean() * 100:.2f}% "
              f"(±{cv_scores.std() * 100:.2f}%)")
        print(f"\n  Classification Report:")
        print(classification_report(
            y_test, preds,
            target_names=["Down", "Up"]
        ))

        # Track best model
        if acc > best_acc:
            best_acc   = acc
            best_model = model
            best_name  = name

    print("\n" + "=" * 60)
    print(f"🏆 Best Model: {best_name} "
          f"({best_acc * 100:.2f}% accuracy)")

    return results, best_model, best_name, \
           X_test, y_test, scaler


# ─────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────

def save_models(results, scaler):
    """Save all trained models to disk"""
    print("\n💾 Saving models...")

    for name, result in results.items():
        # Clean filename
        filename = name.lower().replace(" ", "_")
        path     = f"{MODEL_DIR}/{filename}.pkl"
        joblib.dump(result["model"], path)
        print(f"  ✅ Saved: {path}")

    # Save scaler — needed for Logistic Regression
    scaler_path = f"{MODEL_DIR}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ Saved: {scaler_path}")


# ─────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────

def explain_model(model, X_test, model_name):
    """
    Generate SHAP values to show
    which features matter most
    """
    print(f"\n🔍 Generating SHAP explanation for {model_name}...")

    try:
        # Tree-based explainer for RF and XGBoost
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Handle RF which returns list of arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=FEATURE_COLS,
            show=False,
            plot_type="bar"
        )
        plt.title(
            f"Feature Importance — {model_name}",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        shap_path = f"{OUTPUT_DIR}/shap_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ SHAP plot saved: {shap_path}")

    except Exception as e:
        print(f"  ⚠️  SHAP failed: {e}")


# ─────────────────────────────────────────
# MODEL COMPARISON CHART
# ─────────────────────────────────────────

def plot_model_comparison(results):
    """Bar chart comparing all 3 models"""
    names     = list(results.keys())
    accuracies = [r["accuracy"] for r in results.values()]
    cv_means  = [r["cv_mean"]  for r in results.values()]

    x    = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width,
                   label="Test Accuracy",
                   color=["#2ecc71", "#3498db", "#e74c3c"],
                   alpha=0.8)
    bars2 = ax.bar(x + width/2, cv_means, width,
                   label="CV Mean Accuracy",
                   color=["#27ae60", "#2980b9", "#c0392b"],
                   alpha=0.6)

    # Add value labels on bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )

    ax.set_title(
        "Model Comparison — Test vs CV Accuracy",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Accuracy %")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="red",
               linestyle="--", alpha=0.4,
               label="Random baseline (50%)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Model comparison chart saved: {path}")


# ─────────────────────────────────────────
# PREDICT FOR TODAY
# ─────────────────────────────────────────

def predict_tomorrow(ticker):
    """
    Use saved best model to predict
    next day movement for a ticker
    """
    try:
        # Load best model — try XGBoost first
        model_path = f"{MODEL_DIR}/xgboost.pkl"
        if not os.path.exists(model_path):
            model_path = f"{MODEL_DIR}/random_forest.pkl"

        model = joblib.load(model_path)

        # Get latest data for ticker
        df = load_merged_data(ticker)
        if df.empty:
            return None

        # Use most recent row as features
        latest  = df.iloc[-1]
        features = [[
            latest["news_sentiment"],
            latest["social_sentiment"],
            latest["news_sentiment_lag1"],
            latest["social_sentiment_lag1"],
            latest["volume"],
            latest["prev_day_return"],
            latest["news_count"],
            latest["social_count"]
        ]]

        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()

        return {
            "ticker":     ticker,
            "prediction": "UP 📈" if prediction == 1 else "DOWN 📉",
            "direction":  int(prediction),
            "confidence": round(probability * 100, 1),
            "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"⚠️  Prediction failed for {ticker}: {e}")
        return None


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 — Build feature table
    df = build_feature_table()

    if df.empty:
        print("❌ No data — run collectors + sentiment first")
        sys.exit(1)

    # Step 2 — Train all 3 models
    output = train_models(df)

    if output[0] is None:
        print("❌ Training failed")
        sys.exit(1)

    results, best_model, best_name, \
    X_test, y_test, scaler = output

    # Step 3 — Save all models
    save_models(results, scaler)

    # Step 4 — SHAP explanation
    # Run on tree-based models only
    for name in ["Random Forest", "XGBoost"]:
        if name in results:
            explain_model(
                results[name]["model"],
                X_test,
                name
            )

    # Step 5 — Model comparison chart
    plot_model_comparison(results)

    # Step 6 — Predict tomorrow for all stocks
    print("\n🔮 Tomorrow's Predictions:")
    print("=" * 40)
    for ticker in STOCKS:
        pred = predict_tomorrow(ticker)
        if pred:
            print(f"  {pred['ticker']}: {pred['prediction']} "
                  f"({pred['confidence']}% confidence)")

    print("\n✅ Prediction pipeline complete!")
    print(f"📁 All outputs in: {OUTPUT_DIR}/ and {MODEL_DIR}/")