"""
IT Job Market Analytics — Phase 4: NLP Skill Extraction
Scans each job's description text, matches it against a master list of
known technical skills, and stores the matched skills back into the
SQL database (job_market.db) as a new "skills" column (comma-separated).

This is the same keyword-matching logic that will later be reused on
resumes in Phase 7 (Personal Tools) — one shared skill list, two uses.

SETUP:
    (uses only sqlite3, pandas, re — all already available from earlier phases)

RUN:
    python extract_skills.py

Safe to re-run any time after Phase 3 — it re-reads the "jobs" table
and rewrites the "skills" and "skill_count" columns fresh each time.
"""

import re
import sqlite3
import pandas as pd

DB_FILE = "job_market.db"
TABLE_NAME = "jobs"

# Master skill list — edit/expand this to match your project scope.
# Keep the display name on the left; the regex-safe lowercase form is
# generated automatically below.
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


def build_patterns():
    """Compile a regex for each skill (word-boundary match, case-insensitive)."""
    patterns = {}
    for skill in SKILLS:
        escaped = re.escape(skill)
        patterns[skill] = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    return patterns


def extract_skills(text: str, patterns: dict) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    found = [skill for skill, pattern in patterns.items() if pattern.search(text)]
    return found


def main():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    print(f"Loaded {len(df)} rows from {TABLE_NAME}")

    patterns = build_patterns()
    print(f"Scanning descriptions against {len(SKILLS)} known skills...")

    df["skills_found"] = df["description"].apply(lambda text: extract_skills(text, patterns))
    df["skills"] = df["skills_found"].apply(lambda lst: ", ".join(lst))
    df["skill_count"] = df["skills_found"].apply(len)
    df = df.drop(columns=["skills_found"])

    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    print(f"\nSaved skills back into {DB_FILE} ({TABLE_NAME} table updated)")
    print(f"Rows with at least 1 skill detected: {(df['skill_count'] > 0).sum()} / {len(df)}")

    # Show the most in-demand skills overall — a preview of your dashboard chart
    all_skills = df["skills"].str.split(", ").explode()
    all_skills = all_skills[all_skills != ""]
    top_skills = all_skills.value_counts().head(15)
    print("\n--- Top 15 most in-demand skills across all postings ---")
    print(top_skills.to_string())


if __name__ == "__main__":
    main()
