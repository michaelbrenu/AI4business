"""Generate synthetic education dataset for the analytics dashboard."""

import numpy as np
import pandas as pd
import os

def generate_education_dataset(n_students=500, seed=42):
    """Generate synthetic student performance data with realistic correlations."""
    np.random.seed(seed)

    semesters = ["Fall 2023", "Spring 2024", "Fall 2024", "Spring 2025"]
    courses = ["Mathematics", "Science", "English", "History", "Computer Science"]
    genders = ["Male", "Female", "Non-Binary"]
    parental_edu = ["No Degree", "High School", "Bachelor", "Master", "PhD"]
    income_levels = ["Low", "Medium", "High"]

    parental_edu_score = {"No Degree": 0, "High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
    income_score = {"Low": 0, "Medium": 1, "High": 2}

    rows = []

    for i in range(n_students):
        sid = f"STU-{i+1:04d}"
        gender = np.random.choice(genders, p=[0.45, 0.45, 0.10])
        age = np.random.randint(17, 26)
        parent_ed = np.random.choice(parental_edu, p=[0.10, 0.30, 0.30, 0.20, 0.10])
        income = np.random.choice(income_levels, p=[0.30, 0.45, 0.25])
        prev_gpa = np.clip(np.random.normal(2.8, 0.7), 1.0, 4.0)

        # Each student takes 2-4 semesters
        n_semesters = np.random.randint(2, 5)
        student_semesters = np.random.choice(semesters, size=n_semesters, replace=False)

        for sem in student_semesters:
            course = np.random.choice(courses)

            # Generate correlated features
            base_study = np.clip(np.random.normal(15, 8), 1, 40)
            base_attendance = np.clip(np.random.normal(75, 15), 40, 100)

            # Higher parental education -> slightly more study hours
            study_hours = round(np.clip(base_study + parental_edu_score[parent_ed] * 1.5, 1, 40), 1)
            attendance_rate = round(np.clip(base_attendance + income_score[income] * 3, 40, 100), 1)

            # Grade is a weighted combination
            grade_raw = (
                0.30 * (study_hours / 40 * 100) +
                0.30 * attendance_rate +
                0.20 * (prev_gpa / 4.0 * 100) +
                0.10 * (parental_edu_score[parent_ed] / 4 * 100) +
                0.10 * (income_score[income] / 2 * 100) +
                np.random.normal(0, 8)
            )
            grade = round(np.clip(grade_raw, 0, 100), 1)
            passed = 1 if grade >= 50 else 0

            rows.append({
                "student_id": sid,
                "gender": gender,
                "age": age,
                "study_hours": study_hours,
                "attendance_rate": attendance_rate,
                "previous_grades": round(prev_gpa, 2),
                "parental_education": parent_ed,
                "family_income_level": income,
                "course": course,
                "semester": sem,
                "grade": grade,
                "passed": passed,
            })

    df = pd.DataFrame(rows)

    # Inject ~3% missing values in study_hours, attendance_rate, previous_grades
    for col in ["study_hours", "attendance_rate", "previous_grades"]:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_education_dataset()
    output_path = os.path.join(os.path.dirname(__file__), "education_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows for {df['student_id'].nunique()} students")
    print(f"Saved to {output_path}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nPass rate: {df['passed'].mean()*100:.1f}%")
    print(f"Mean grade: {df['grade'].mean():.1f}")
