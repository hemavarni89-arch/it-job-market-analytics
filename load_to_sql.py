"""
IT Job Market Analytics — Phase 3: SQL Storage
Loads job_postings_clean.csv (from Phase 2) into a SQLite database table
called "jobs". SQLite needs no separate server — the whole database is
just one file (job_market.db) sitting in your project folder.

SETUP:
    (sqlite3 is built into Python — nothing extra to install)

RUN:
    python load_to_sql.py

Safe to run repeatedly: each run rebuilds the "jobs" table fresh from
the latest job_postings_clean.csv, so always re-run Phase 2 first if
you've collected new data.
"""

import sqlite3
import pandas as pd

INPUT_FILE = "job_postings_clean.csv"
DB_FILE = "job_market.db"
TABLE_NAME = "jobs"


def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    return df


def write_to_sql(df: pd.DataFrame):
    conn = sqlite3.connect(DB_FILE)
    # if_exists="replace" rebuilds the table fresh each run, so re-running
    # this script after collecting more data always reflects the latest CSV
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} rows into table '{TABLE_NAME}' inside {DB_FILE}")


def run_sample_queries():
    """A few example queries to confirm the data loaded correctly and to
    show the kind of aggregation your Streamlit dashboard will later run."""
    conn = sqlite3.connect(DB_FILE)

    print("\n--- Top 5 companies by posting count ---")
    q1 = f"""
        SELECT company, COUNT(*) AS postings
        FROM {TABLE_NAME}
        GROUP BY company
        ORDER BY postings DESC
        LIMIT 5
    """
    print(pd.read_sql(q1, conn).to_string(index=False))

    print("\n--- Top 5 locations by posting count ---")
    q2 = f"""
        SELECT location_clean, COUNT(*) AS postings
        FROM {TABLE_NAME}
        GROUP BY location_clean
        ORDER BY postings DESC
        LIMIT 5
    """
    print(pd.read_sql(q2, conn).to_string(index=False))

    print("\n--- Average salary (where available) by role_query ---")
    q3 = f"""
        SELECT role_query,
               ROUND(AVG(salary_min)) AS avg_min_salary,
               ROUND(AVG(salary_max)) AS avg_max_salary,
               COUNT(*) AS postings
        FROM {TABLE_NAME}
        WHERE has_salary_info = 1
        GROUP BY role_query
        ORDER BY postings DESC
    """
    print(pd.read_sql(q3, conn).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    df = load_dataframe()
    write_to_sql(df)
    run_sample_queries()
