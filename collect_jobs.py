"""
IT Job Market Analytics — Data Collection Script
Collects IT job postings from the Adzuna API (aggregates listings from
Indeed, Naukri, and other job boards) and saves them to a CSV file.

SETUP (do this once):
1. Sign up free at https://developer.adzuna.com/ and get your app_id + app_key
2. Create a file named ".env" in the same folder as this script, containing:
       ADZUNA_APP_ID=your_app_id_here
       ADZUNA_APP_KEY=your_app_key_here
3. Install dependencies:
       pip install requests pandas python-dotenv

RUN:
    python collect_jobs.py

Run this once a day (or more) — each run appends a fresh snapshot with
today's date, which is what lets you track posting/reposting trends
over time for your Estimated Hiring Demand Score.
"""

import os
import time
import requests
import pandas as pd
from datetime import date
from dotenv import load_dotenv

# ---------- CONFIG ----------
load_dotenv()
APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")
COUNTRY = "in"                      # India. Use "gb", "us", etc. for other countries
RESULTS_PER_PAGE = 20               # Adzuna's max per page
PAGES_PER_ROLE = 10                 # 10 pages x 20 = ~200 postings per role per run
OUTPUT_FILE = "job_postings.csv"    # all collected data accumulates here

# Edit this list to match the roles your project targets
ROLES = [
    "Data Analyst",
    "Data Scientist",
    "Software Engineer",
    "Full Stack Developer",
    "DevOps Engineer",
    "QA Engineer",
    "UI UX Designer",
    "Cloud Engineer",
    "Machine Learning Engineer",
    "Business Analyst",
    "Backend Developer",
    "Frontend Developer",
    "Cybersecurity Analyst",
    "Database Administrator",
    "Product Manager",
    "System Administrator",
    "Network Engineer",
    "Mobile App Developer",
]

BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"


def fetch_jobs(role: str, page: int) -> list[dict]:
    """Fetch one page of results for a given role."""
    url = BASE_URL.format(country=COUNTRY, page=page)
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": RESULTS_PER_PAGE,
        "what": role,
        "content-type": "application/json",
    }
    response = requests.get(url, params=params, timeout=15)
    if response.status_code != 200:
        print(f"  [warn] {role} page {page} -> HTTP {response.status_code}")
        return []
    return response.json().get("results", [])


def parse_job(job: dict, role_query: str) -> dict:
    """Pull out only the fields we need from Adzuna's raw job object."""
    return {
        "date_collected": date.today().isoformat(),
        "role_query": role_query,
        "job_id": job.get("id"),
        "title": job.get("title", "").strip(),
        "company": (job.get("company") or {}).get("display_name", ""),
        "location": (job.get("location") or {}).get("display_name", ""),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "contract_type": job.get("contract_type", ""),
        "category": (job.get("category") or {}).get("label", ""),
        "description": job.get("description", ""),
        "posted_date": job.get("created", ""),
        "url": job.get("redirect_url", ""),
    }


def collect_all() -> pd.DataFrame:
    all_jobs = []
    for role in ROLES:
        print(f"Collecting: {role}")
        for page in range(1, PAGES_PER_ROLE + 1):
            results = fetch_jobs(role, page)
            if not results:
                break
            for job in results:
                all_jobs.append(parse_job(job, role))
            time.sleep(0.5)  # be polite to the API — avoid rate-limit errors
    return pd.DataFrame(all_jobs)


def save(df: pd.DataFrame):
    if df.empty:
        print("No data collected — check your API keys and internet connection.")
        return

    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        combined = pd.concat([existing, df], ignore_index=True)
        # avoid exact duplicate rows from the same job on the same day
        combined.drop_duplicates(subset=["job_id", "date_collected"], inplace=True)
    else:
        combined = df

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} new rows. Total dataset now: {len(combined)} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    if not APP_ID or not APP_KEY:
        raise SystemExit(
            "Missing API credentials. Create a .env file with ADZUNA_APP_ID "
            "and ADZUNA_APP_KEY — see the setup notes at the top of this file."
        )
    df = collect_all()
    save(df)
