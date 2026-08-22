"""
IT Job Market Analytics — Phase 8c: Salary & Forecast page
Second personal-tools-style page: lets the user pick a role, location,
and experience level to get a predicted salary from the trained model,
and shows the hiring trend forecast chart per role.

Folder structure required:
    job-market-project/
    └── pages/
        ├── 1_Personal_Tools.py
        └── 2_Salary_And_Forecast.py   <- this file
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

DB_FILE = "job_market.db"
MODEL_FILE = "salary_model.pkl"

st.set_page_config(page_title="Salary & Forecast", layout="wide")
st.title("Salary Prediction & Hiring Trend Forecast")


@st.cache_data(ttl=300)
def load_jobs():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_forecast():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("SELECT * FROM trend_forecast", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return None


jobs_df = load_jobs()
forecast_df = load_forecast()
model_bundle = load_model()

# ---------- Salary Prediction ----------
st.subheader("Salary Prediction")

if model_bundle is None:
    st.warning("No trained model found yet. Run `python train_salary_model.py` first, then reload this page.")
else:
    model = model_bundle["model"]
    role_encoder = model_bundle["role_encoder"]
    location_encoder = model_bundle["location_encoder"]

    c1, c2, c3 = st.columns(3)
    role = c1.selectbox("Role", sorted(role_encoder.classes_))
    location = c2.selectbox("Location", sorted(location_encoder.classes_))
    experience_label = c3.selectbox(
        "Experience level",
        ["0-1 years (Fresher)", "1-3 years", "3-5 years", "5+ years"]
    )
    experience_map = {"0-1 years (Fresher)": 0, "1-3 years": 1, "3-5 years": 2, "5+ years": 3}
    experience_level = experience_map[experience_label]

    skill_count = st.slider("Number of relevant skills you have", 0, 15, 5)

    if st.button("Predict Salary", type="primary"):
        role_encoded = role_encoder.transform([role])[0]
        location_encoded = location_encoder.transform([location])[0]
        X_input = pd.DataFrame([{
            "role_encoded": role_encoded,
            "location_encoded": location_encoded,
            "experience_level": experience_level,
            "skill_count": skill_count,
        }])
        predicted = model.predict(X_input)[0]
        low = predicted * 0.85
        high = predicted * 1.15

        st.success(f"**Predicted salary: ₹{predicted:,.0f} / year**")
        st.caption(f"Estimated range: ₹{low:,.0f} – ₹{high:,.0f}, based on similar postings in the dataset.")

st.divider()

# ---------- Trend Forecasting ----------
st.subheader("Hiring Trend Forecast")

if forecast_df.empty:
    st.info("No forecast data yet. Run `python forecast_trend.py` first, then reload this page.")
else:
    roles_available = sorted(forecast_df["role_query"].unique())
    selected_role = st.selectbox("Select a role to view its trend", roles_available)

    role_data = forecast_df[forecast_df["role_query"] == selected_role].sort_values("week")

    fig = px.line(
        role_data, x="week", y="predicted_postings", color="type",
        markers=True,
        color_discrete_map={"actual": "#38D6C4", "forecast": "#FF9F45"},
        labels={"predicted_postings": "Postings per week", "week": "Week"}
    )
    fig.update_traces(selector=dict(name="forecast"), line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Solid line: actual weekly posting counts (from original posting dates). "
               "Dashed line: projected postings for the next 4 weeks using a linear trend.")
