"""
IT Job Market Analytics — Phase 2: Data Cleaning
Reads job_postings.csv (from Phase 1) and produces a cleaned version:
- removes duplicate postings
- handles missing values
- standardizes text (casing, whitespace, location names)

SETUP:
    pip install pandas

RUN:
    python clean_data.py

Run this any time after collecting new data — it always re-reads the
full job_postings.csv and regenerates job_postings_clean.csv from scratch,
so it's safe to run repeatedly.
"""

import pandas as pd
import re

INPUT_FILE = "job_postings.csv"
OUTPUT_FILE = "job_postings_clean.csv"

# Common location name variants -> standardized name
LOCATION_MAP = {
    "bengaluru": "Bangalore",
    "bangalore": "Bangalore",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "gurugram": "Gurgaon",
    "gurgaon": "Gurgaon",
    "new delhi": "Delhi",
    "delhi": "Delhi",
    "trivandrum": "Thiruvananthapuram",
    "cochin": "Kochi",
}


def clean_text(value: str) -> str:
    """Trim whitespace, collapse multiple spaces, keep original casing for readability."""
    if pd.isna(value):
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def standardize_location(location: str) -> str:
    """Extract the first city-like token and map common variants to one name."""
    if pd.isna(location) or not str(location).strip():
        return "Not specified"
    first_part = str(location).split(",")[0].strip()
    key = first_part.lower()
    return LOCATION_MAP.get(key, first_part.title())


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} raw rows from {INPUT_FILE}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    # same job_id appearing on different dates is a valid re-posting signal for
    # your demand score, so we only drop TRUE duplicates: same job_id + same day
    df = df.drop_duplicates(subset=["job_id", "date_collected"])
    print(f"Removed {before - len(df)} exact duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df["title"] = df["title"].apply(clean_text)
    df["company"] = df["company"].apply(clean_text)
    df["company"] = df["company"].replace("", "Not specified")
    df["description"] = df["description"].apply(clean_text)

    # salary: leave blank cells as NaN (don't invent numbers), but flag it
    df["salary_min"] = pd.to_numeric(df["salary_min"], errors="coerce")
    df["salary_max"] = pd.to_numeric(df["salary_max"], errors="coerce")
    df["has_salary_info"] = df["salary_min"].notna() | df["salary_max"].notna()

    df["contract_type"] = df["contract_type"].apply(clean_text)
    df["contract_type"] = df["contract_type"].replace("", "Not specified")

    # drop rows that are missing the essentials — nothing usable without these
    before = len(df)
    df = df.dropna(subset=["job_id", "title"])
    df = df[df["title"] != ""]
    print(f"Dropped {before - len(df)} rows missing essential fields (job_id/title)")
    return df


def standardize_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["location_clean"] = df["location"].apply(standardize_location)
    df["title"] = df["title"].str.title()
    return df


def save(df: pd.DataFrame):
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved cleaned dataset: {len(df)} rows -> {OUTPUT_FILE}")
    print("\nQuick summary:")
    print(f"  Unique companies: {df['company'].nunique()}")
    print(f"  Unique locations: {df['location_clean'].nunique()}")
    print(f"  Rows with salary info: {df['has_salary_info'].sum()} / {len(df)}")


if __name__ == "__main__":
    df = load_data()
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = standardize_fields(df)
    save(df)
