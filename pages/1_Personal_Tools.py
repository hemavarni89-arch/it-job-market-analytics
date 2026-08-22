"""
IT Job Market Analytics — Phase 7: Personal Tools
Streamlit multipage app page: resume upload -> extract skills -> compare
against every job in job_market.db -> show ranked recommendations and
a skill gap summary.

This file must live inside a folder named "pages" next to your main
app.py, so Streamlit automatically shows it as a second page in the
sidebar. Streamlit's convention: pages/1_Personal_Tools.py

SETUP:
    pip install pdfplumber python-docx scikit-learn

Folder structure required:
    job-market-project/
    ├── app.py
    ├── job_market.db
    └── pages/
        └── 1_Personal_Tools.py   <- this file
"""

import re
import sqlite3
import pandas as pd
import streamlit as st
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_FILE = "job_market.db"

# Same master skill list used in Phase 4 (extract_skills.py) — keep these
# two lists in sync so resume skills and job skills are comparable.
SKILLS = [
    "Python", "SQL", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go",
    "R", "Scala", "Kotlin", "Swift", "PHP",
    "React", "Angular", "Vue", "Node.js", "Django", "Flask", "Spring Boot",
    "HTML", "CSS", "REST API", "GraphQL",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "CI/CD",
    "Terraform", "Linux", "Git", "DevOps",
    "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Power BI", "Tableau", "Excel", "Data Analysis", "Data Visualization",
    "ETL", "Data Warehousing", "Big Data", "Spark", "Hadoop",
    "MySQL", "PostgreSQL", "MongoDB", "Oracle", "NoSQL", "Redis",
    "Android", "iOS", "Flutter", "React Native",
    "Figma", "Adobe XD", "UI Design", "UX Research", "Wireframing",
    "Selenium", "JUnit", "Manual Testing", "Automation Testing",
    "Agile", "Scrum", "JIRA", "Project Management",
]


@st.cache_resource
def build_patterns():
    return {s: re.compile(rf"\b{re.escape(s)}\b", re.IGNORECASE) for s in SKILLS}


def extract_skills_from_text(text: str, patterns: dict) -> list[str]:
    if not text or not text.strip():
        return []
    return [s for s, p in patterns.items() if p.search(text)]


def extract_text_from_pdf(file) -> str:
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(file) -> str:
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)


@st.cache_data(ttl=300)
def load_jobs() -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()
    return df


st.set_page_config(page_title="Personal Tools", layout="wide")
st.title("Personal Tools")
st.caption("Upload your resume to see your extracted skills, skill gaps, and ranked job recommendations "
           "based on the current job dataset.")

patterns = build_patterns()
jobs_df = load_jobs()

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    with st.spinner("Reading your resume..."):
        if uploaded_file.name.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = extract_text_from_docx(uploaded_file)

    resume_skills = extract_skills_from_text(resume_text, patterns)

    if not resume_text.strip():
        st.error("Couldn't extract any text from this file — try a different resume file "
                  "(make sure it's not a scanned image PDF).")
    else:
        st.success(f"Resume processed — {len(resume_skills)} known skills detected.")

        col_a, col_b = st.columns([1, 1.5])

        with col_a:
            st.subheader("Resume Analyzer")
            st.markdown("**Extracted skills:**")
            if resume_skills:
                st.markdown(" ".join(f"`{s}`" for s in resume_skills))
            else:
                st.info("No skills from our master list were found in this resume.")

        # ---------- Matching: TF-IDF + cosine similarity, resume text vs job descriptions ----------
        with st.spinner("Comparing your resume against current job postings..."):
            valid_jobs = jobs_df[jobs_df["description"].notna() & (jobs_df["description"] != "")].copy()
            corpus = [resume_text] + valid_jobs["description"].tolist()
            vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            resume_vector = tfidf_matrix[0:1]
            job_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(resume_vector, job_vectors).flatten()
            valid_jobs["match_pct"] = (similarities * 100).round(1)

        top_matches = valid_jobs.sort_values("match_pct", ascending=False).head(10)

        with col_b:
            st.subheader("Job Recommendations")
            for _, row in top_matches.iterrows():
                job_skills = [s.strip() for s in str(row.get("skills", "")).split(",") if s.strip()]
                missing = [s for s in job_skills if s not in resume_skills]
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**{row['title']}** — {row['company']}")
                    c1.caption(f"{row['location_clean']}")
                    c2.metric("Match", f"{row['match_pct']:.0f}%")
                    if missing:
                        st.caption(f"Missing skills: {', '.join(missing[:5])}")

        st.divider()

        # ---------- Skill Gap Analyzer: aggregate missing skills across top matches ----------
        st.subheader("Skill Gap Analyzer")
        all_missing = []
        for _, row in top_matches.iterrows():
            job_skills = [s.strip() for s in str(row.get("skills", "")).split(",") if s.strip()]
            all_missing.extend([s for s in job_skills if s not in resume_skills])

        if all_missing:
            gap_counts = pd.Series(all_missing).value_counts().head(10)
            gap_df = gap_counts.reset_index()
            gap_df.columns = ["skill", "appears_in_top_matches"]
            st.markdown("Skills that appear most often in your top-matched jobs, but aren't on your resume:")
            st.dataframe(gap_df, use_container_width=True, hide_index=True)
        else:
            st.success("No major skill gaps detected against your top matches — nice.")
else:
    st.info("Upload a resume above to get started.")
