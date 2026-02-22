"""Generate synthetic health/patient dataset for testing the dashboard."""

import numpy as np
import pandas as pd
import os


def generate_health_dataset(n_patients=600, seed=42):
    """Generate synthetic patient health data with realistic correlations."""
    np.random.seed(seed)

    departments = ["Cardiology", "Orthopedics", "Neurology", "Oncology", "Pediatrics"]
    genders = ["Male", "Female"]
    blood_types = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
    admission_types = ["Emergency", "Elective", "Urgent"]
    quarters = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
    insurance = ["Private", "Medicare", "Medicaid", "Uninsured"]
    discharge_status = ["Recovered", "Improved", "Transferred", "Deceased"]

    rows = []

    for i in range(n_patients):
        pid = f"PAT-{i+1:05d}"
        gender = np.random.choice(genders, p=[0.48, 0.52])
        age = int(np.clip(np.random.normal(55, 18), 1, 95))
        blood = np.random.choice(blood_types, p=[0.30, 0.06, 0.09, 0.02, 0.38, 0.07, 0.06, 0.02])
        dept = np.random.choice(departments, p=[0.25, 0.20, 0.15, 0.20, 0.20])
        admit_type = np.random.choice(admission_types, p=[0.35, 0.40, 0.25])
        quarter = np.random.choice(quarters)
        ins = np.random.choice(insurance, p=[0.40, 0.25, 0.20, 0.15])

        # Correlated health metrics
        bmi = round(np.clip(np.random.normal(27, 5), 15, 50), 1)
        blood_pressure_sys = int(np.clip(
            90 + age * 0.5 + bmi * 0.8 + np.random.normal(0, 12), 80, 200
        ))
        heart_rate = int(np.clip(
            80 - age * 0.1 + bmi * 0.3 + np.random.normal(0, 10), 50, 130
        ))

        # Length of stay correlated with age, admission type, and department
        base_los = 3
        if admit_type == "Emergency":
            base_los += 2
        if dept in ["Oncology", "Cardiology"]:
            base_los += 2
        if age > 65:
            base_los += 1.5
        length_of_stay = max(1, int(np.clip(base_los + np.random.normal(0, 2), 1, 30)))

        # Treatment cost correlated with LOS, department, insurance
        base_cost = length_of_stay * 1200
        if dept == "Oncology":
            base_cost *= 1.5
        elif dept == "Cardiology":
            base_cost *= 1.3
        treatment_cost = round(np.clip(base_cost + np.random.normal(0, 1500), 500, 80000), 2)

        # Satisfaction score (1-10) inversely related to LOS and cost
        satisfaction = round(np.clip(
            8.5 - length_of_stay * 0.15 - (treatment_cost / 50000) + np.random.normal(0, 1.2),
            1, 10
        ), 1)

        # Readmission (binary) - higher risk for older, longer stays, emergency
        readmit_prob = 0.05
        if age > 65:
            readmit_prob += 0.08
        if length_of_stay > 7:
            readmit_prob += 0.10
        if admit_type == "Emergency":
            readmit_prob += 0.05
        if ins == "Uninsured":
            readmit_prob += 0.07
        readmitted = 1 if np.random.random() < readmit_prob else 0

        # Discharge status
        if readmitted:
            status = np.random.choice(discharge_status, p=[0.30, 0.50, 0.15, 0.05])
        else:
            status = np.random.choice(discharge_status, p=[0.55, 0.35, 0.08, 0.02])

        rows.append({
            "patient_id": pid,
            "gender": gender,
            "age": age,
            "blood_type": blood,
            "bmi": bmi,
            "blood_pressure_systolic": blood_pressure_sys,
            "heart_rate": heart_rate,
            "department": dept,
            "admission_type": admit_type,
            "insurance_type": ins,
            "quarter": quarter,
            "length_of_stay": length_of_stay,
            "treatment_cost": treatment_cost,
            "satisfaction_score": satisfaction,
            "readmitted": readmitted,
            "discharge_status": status,
        })

    df = pd.DataFrame(rows)

    # Inject ~3% missing values
    for col in ["bmi", "blood_pressure_systolic", "heart_rate", "satisfaction_score"]:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    # Inject some whitespace/casing inconsistencies for cleaning demo
    casing_mask = np.random.random(len(df)) < 0.02
    df.loc[casing_mask, "department"] = df.loc[casing_mask, "department"].str.upper()

    space_mask = np.random.random(len(df)) < 0.02
    df.loc[space_mask, "admission_type"] = df.loc[space_mask, "admission_type"] + "  "

    return df


if __name__ == "__main__":
    df = generate_health_dataset()
    output_path = os.path.join(os.path.dirname(__file__), "health_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} patient records")
    print(f"Saved to {output_path}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nReadmission rate: {df['readmitted'].mean()*100:.1f}%")
    print(f"Avg length of stay: {df['length_of_stay'].mean():.1f} days")
    print(f"Avg treatment cost: ${df['treatment_cost'].mean():,.0f}")
    print(f"Avg satisfaction: {df['satisfaction_score'].mean():.1f}/10")
