"""
IT Job Market Analytics — Phase 6: Streamlit Market Dashboard
Reads directly from job_market.db (built in Phases 1-5) and shows
interactive charts: in-demand skills, top companies, top locations,
salary benchmarks, work-mode/experience breakdown, and the Estimated
Hiring Demand Score by role.

SETUP:
    pip install streamlit plotly pandas

RUN:
    streamlit run app.py

This opens automatically in your browser (usually http://localhost:8501).
Leave the Command Prompt window open while using it — closing it stops
the app. Press Ctrl+C in that window to stop the app manually.
"""

import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

DB_FILE = "job_market.db"

st.set_page_config(page_title="IT Job Market Analytics", layout="wide")


@st.cache_data(ttl=300)
def load_jobs() -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_demand_scores() -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("SELECT * FROM demand_scores", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


df = load_jobs()
scores_df = load_demand_scores()

st.title("IT Job Market Analytics")
st.caption("Live view built from Adzuna-sourced job postings, cleaned and processed through the project's data pipeline.")

# ---------- KPI row ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Postings Tracked", f"{len(df):,}")
col2.metric("Unique Companies", f"{df['company'].nunique():,}")
col3.metric("Unique Locations", f"{df['location_clean'].nunique():,}")
col4.metric("Postings with Salary Info", f"{int(df['has_salary_info'].sum()):,}")

st.divider()

# ---------- Row 1: Skills + Work mode-ish (contract type) ----------
c1, c2 = st.columns([1.3, 1])

with c1:
    st.subheader("In-Demand Technical Skills")
    all_skills = df["skills"].dropna().str.split(", ").explode()
    all_skills = all_skills[all_skills != ""]
    top_skills = all_skills.value_counts().head(12).reset_index()
    top_skills.columns = ["skill", "count"]
    fig = px.bar(top_skills, x="count", y="skill", orientation="h",
                 color_discrete_sequence=["#38D6C4"])
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Postings by Contract Type")
    ct = df["contract_type"].value_counts().reset_index()
    ct.columns = ["contract_type", "count"]
    fig = px.pie(ct, names="contract_type", values="count", hole=0.55)
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Row 2: Companies + Locations ----------
c3, c4 = st.columns(2)

with c3:
    st.subheader("Top Hiring Companies")
    top_companies = df["company"].value_counts().head(10).reset_index()
    top_companies.columns = ["company", "postings"]
    fig = px.bar(top_companies, x="postings", y="company", orientation="h",
                 color_discrete_sequence=["#6E8CFF"])
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("Top Hiring Locations")
    top_locations = df["location_clean"].value_counts().head(10).reset_index()
    top_locations.columns = ["location", "postings"]
    fig = px.bar(top_locations, x="postings", y="location", orientation="h",
                 color_discrete_sequence=["#C58CFF"])
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Row 3: Salary benchmarks ----------
st.subheader("Salary Benchmarks by Role")
salary_df = df[df["has_salary_info"] == 1]
if not salary_df.empty:
    salary_summary = (
        salary_df.groupby("role_query")[["salary_min", "salary_max"]]
        .mean().round(0).reset_index()
    )
    salary_summary = salary_summary.sort_values("salary_max", ascending=False)
    fig = px.bar(
        salary_summary, x="role_query", y=["salary_min", "salary_max"],
        barmode="group", labels={"value": "Salary (INR/year)", "role_query": "Role"},
        color_discrete_sequence=["#1E2E47", "#38D6C4"]
    )
    fig.update_layout(height=420, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No salary data available yet.")

# ---------- Row 4: Estimated Hiring Demand Score ----------
st.subheader("Estimated Hiring Demand Score by Role")
if not scores_df.empty:
    scores_sorted = scores_df.sort_values("demand_score", ascending=False)
    fig = px.bar(scores_sorted, x="role", y="demand_score",
                 color="demand_score", color_continuous_scale="Teal")
    fig.update_layout(height=420, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Score = 50% posting frequency + 30% reposting pattern + 20% recency (0-100 scale). "
               "An estimate of relative market demand, not actual hiring figures.")
else:
    st.info("Run demand_score.py first to populate this chart.")

st.divider()
st.caption("Data source: Adzuna API (aggregates listings from Indeed, Naukri, and other job boards). "
           "Sample reflects publicly available postings, not a full census of the job market.")
