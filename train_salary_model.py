"""
IT Job Market Analytics — Phase 8a: Salary Prediction Model (training)
Trains a Random Forest regression model on your collected job data to
predict expected salary from role, location, and skill count. Saves
the trained model + encoders to disk so the Streamlit app can load it
instantly without retraining every time.

SETUP:
    pip install scikit-learn joblib

RUN:
    python train_salary_model.py

Re-run this any time after collecting more data (more rows -> better
model). It always retrains from scratch using whatever is currently
in job_market.db.
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DB_FILE = "job_market.db"
MODEL_FILE = "salary_model.pkl"

# A rough experience-level estimate: 0=fresher/unspecified, 1=1-3yr, 2=3-5yr, 3=5+yr
# extracted from the description text with simple keyword rules.
def estimate_experience(text: str) -> int:
    if not isinstance(text, str):
        return 0
    text = text.lower()
    import re
    match = re.search(r"(\d+)\s*(?:\+)?\s*(?:to\s*\d+\s*)?years?", text)
    if match:
        years = int(match.group(1))
        if years >= 5:
            return 3
        elif years >= 3:
            return 2
        elif years >= 1:
            return 1
    if "fresher" in text or "entry level" in text or "0-1" in text:
        return 0
    return 0


def main():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()
    print(f"Loaded {len(df)} rows")

    # Use average of salary_min/salary_max as the training target
    df = df[df["has_salary_info"] == 1].copy()
    df["salary_avg"] = df[["salary_min", "salary_max"]].mean(axis=1)
    df = df.dropna(subset=["salary_avg", "role_query", "location_clean"])
    print(f"{len(df)} rows have usable salary data for training")

    if len(df) < 30:
        print("Not enough salary-labeled rows to train a reliable model yet. "
              "Keep collecting more data and re-run this script later.")
        return

    df["experience_level"] = df["description"].apply(estimate_experience)
    df["skill_count"] = df["skill_count"].fillna(0)

    # Encode categorical fields
    role_encoder = LabelEncoder()
    location_encoder = LabelEncoder()
    df["role_encoded"] = role_encoder.fit_transform(df["role_query"])
    df["location_encoded"] = location_encoder.fit_transform(df["location_clean"])

    features = ["role_encoded", "location_encoded", "experience_level", "skill_count"]
    X = df[features]
    y = df["salary_avg"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\nModel performance on held-out test data:")
    print(f"  Mean Absolute Error: ₹{mae:,.0f}")
    print(f"  R² score: {r2:.3f}")
    print("(R² closer to 1.0 is better; with a modest dataset, 0.3-0.6 is a reasonable "
          "range to report honestly in your project documentation.)")

    joblib.dump({
        "model": model,
        "role_encoder": role_encoder,
        "location_encoder": location_encoder,
        "features": features,
    }, MODEL_FILE)
    print(f"\nSaved trained model -> {MODEL_FILE}")


if __name__ == "__main__":
    main()
