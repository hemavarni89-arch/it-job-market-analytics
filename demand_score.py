"""
IT Job Market Analytics — Phase 5: Estimated Hiring Demand Score
Calculates a 0-100 demand score per role, based on:
  - Posting frequency  : how many postings exist for that role
  - Reposting pattern   : how often the same job_id reappears across
                          different date_collected snapshots (a proxy
                          for a role staying open / being re-advertised)
  - Recency             : how many postings were created recently
                          (based on Adzuna's own "posted_date" field)

Since official recruitment stats aren't public, this score is an
ESTIMATE of relative market demand, not an actual hiring count.

SETUP:
    (uses only sqlite3, pandas — already available from earlier phases)

RUN:
    python demand_score.py

Re-run this after each new daily collection run (Phase 1) followed by
Phase 2/3/4 — the more daily snapshots you have, the more meaningful
the reposting signal becomes. With only one day of data collected so
far, the reposting component will show as 0 for everyone; that's
expected and will fill in as you collect over more days.
"""

import sqlite3
import pandas as pd

DB_FILE = "job_market.db"
TABLE_NAME = "jobs"
SCORE_TABLE = "demand_scores"

# Weights — documented clearly since your report/viva should explain this formula
WEIGHT_FREQUENCY = 0.5
WEIGHT_REPOSTING = 0.3
WEIGHT_RECENCY = 0.2


def normalize(series: pd.Series) -> pd.Series:
    """Scale a numeric series to 0-100. Flat series (no variation) -> all zeros."""
    min_v, max_v = series.min(), series.max()
    if max_v == min_v:
        return pd.Series([0] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v) * 100


def main():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    print(f"Loaded {len(df)} rows from {TABLE_NAME}")

    n_snapshot_days = df["date_collected"].nunique()
    print(f"Data currently spans {n_snapshot_days} collection day(s).")
    if n_snapshot_days < 3:
        print("Note: reposting signal is most meaningful after several days "
              "of daily collection — keep running collect_jobs.py daily.")

    # --- 1. Posting frequency per role ---
    freq = df.groupby("role_query").size().rename("posting_count")

    # --- 2. Reposting pattern: how many times, on average, the same job_id
    #        reappears across different snapshot days, per role ---
    repost = (
        df.groupby(["role_query", "job_id"])["date_collected"]
        .nunique()
        .reset_index(name="days_seen")
    )
    repost_avg = repost.groupby("role_query")["days_seen"].mean().rename("avg_repost")

    # --- 3. Recency: proportion of a role's postings created in the last 7 days ---
    df["posted_date_parsed"] = pd.to_datetime(df["posted_date"], errors="coerce", utc=True)
    latest_date = df["posted_date_parsed"].max()
    df["is_recent"] = (latest_date - df["posted_date_parsed"]).dt.days <= 7
    recency = df.groupby("role_query")["is_recent"].mean().rename("recency_ratio")

    # --- Combine ---
    scores = pd.concat([freq, repost_avg, recency], axis=1).fillna(0)
    scores["freq_score"] = normalize(scores["posting_count"])
    scores["repost_score"] = normalize(scores["avg_repost"])
    scores["recency_score"] = normalize(scores["recency_ratio"])

    scores["demand_score"] = (
        scores["freq_score"] * WEIGHT_FREQUENCY
        + scores["repost_score"] * WEIGHT_REPOSTING
        + scores["recency_score"] * WEIGHT_RECENCY
    ).round(1)

    scores = scores.sort_values("demand_score", ascending=False).reset_index()
    scores = scores.rename(columns={"role_query": "role"})

    # Save to its own table so the Streamlit dashboard can read it directly
    scores.to_sql(SCORE_TABLE, conn, if_exists="replace", index=False)
    conn.close()

    print(f"\nSaved demand scores into table '{SCORE_TABLE}' inside {DB_FILE}")
    print("\n--- Estimated Hiring Demand Score by role ---")
    print(scores[["role", "posting_count", "avg_repost", "recency_ratio", "demand_score"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
