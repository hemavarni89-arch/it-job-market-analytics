"""
IT Job Market Analytics — Phase 8b: Trend Forecasting Model
Uses each posting's ORIGINAL posted_date (not date_collected) to build
a weekly posting-volume trend per role, then projects it forward using
a simple linear trend line. This works even from a single collection
run, because Adzuna's posted_date field already spans several weeks
of history for existing listings.

For a more accurate forecast later, re-run this after you've collected
across several different days too (date_collected) — for now this
gives a reasonable first estimate from posting recency patterns alone.

SETUP:
    pip install scikit-learn

RUN:
    python forecast_trend.py
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

DB_FILE = "job_market.db"
FORECAST_TABLE = "trend_forecast"
WEEKS_AHEAD = 4


def main():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT role_query, posted_date FROM jobs", conn)

    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["posted_date"])
    df["week"] = df["posted_date"].dt.to_period("W").apply(lambda p: p.start_time)

    weekly_counts = df.groupby(["role_query", "week"]).size().reset_index(name="postings")

    all_forecasts = []
    for role, group in weekly_counts.groupby("role_query"):
        group = group.sort_values("week")
        if len(group) < 3:
            # not enough historical weeks for this role to fit a trend line
            continue

        group = group.reset_index(drop=True)
        group["week_index"] = np.arange(len(group))

        X = group[["week_index"]]
        y = group["postings"]
        model = LinearRegression()
        model.fit(X, y)

        last_week = group["week"].max()
        last_index = group["week_index"].max()
        future_indices = np.arange(last_index + 1, last_index + 1 + WEEKS_AHEAD)
        future_weeks = [last_week + pd.Timedelta(weeks=i) for i in range(1, WEEKS_AHEAD + 1)]
        future_preds = model.predict(future_indices.reshape(-1, 1))
        future_preds = np.maximum(future_preds, 0)  # postings can't go negative

        for wk, pred in zip(future_weeks, future_preds):
            all_forecasts.append({
                "role_query": role,
                "week": wk,
                "predicted_postings": round(float(pred), 1),
                "type": "forecast",
            })
        for _, row in group.iterrows():
            all_forecasts.append({
                "role_query": role,
                "week": row["week"],
                "predicted_postings": float(row["postings"]),
                "type": "actual",
            })

    if not all_forecasts:
        print("Not enough historical weekly spread in posted_date to build a trend yet. "
              "This usually resolves once your dataset covers a wider range of original "
              "posting dates.")
        conn.close()
        return

    forecast_df = pd.DataFrame(all_forecasts)
    forecast_df["week"] = forecast_df["week"].astype(str)
    forecast_df.to_sql(FORECAST_TABLE, conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved forecast data into table '{FORECAST_TABLE}' inside {DB_FILE}")
    print(f"Roles with enough history to forecast: {forecast_df['role_query'].nunique()}")
    print("\nSample (first role forecasted):")
    sample_role = forecast_df["role_query"].iloc[0]
    print(forecast_df[forecast_df["role_query"] == sample_role].to_string(index=False))


if __name__ == "__main__":
    main()
