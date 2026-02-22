# AI-Powered Data Analysis & Visualization Dashboard

## Complete Application Guide

---

## Overview

This application is an **AI-assisted analytics dashboard** that transforms raw CSV/Excel data into actionable insights through a guided 7-step wizard. It combines data profiling, automated cleaning, interactive visualizations, machine learning predictions, and GPT-powered narrative generation — all in a browser-based interface.

**Tech Stack:** Python, Streamlit, Pandas, Plotly, scikit-learn, OpenAI API, fpdf2

---

## Project Structure

```
c:/DevOps/Ai 4 Business/
|
|-- app.py                          # Main application (915 lines) - the wizard UI
|-- requirements.txt                # Python dependencies
|-- steps.md                        # This document
|
|-- data/
|   |-- generate_dataset.py         # Generates sample education dataset
|   |-- generate_health_data.py     # Generates sample health dataset
|   |-- education_data.csv          # 1,485 rows - student performance data
|   |-- health_data.csv             # 600 rows - patient health records
|
|-- src/
|   |-- __init__.py                 # Package marker
|   |-- data_loader.py              # File upload handling (CSV/Excel)
|   |-- data_profiler.py            # Automatic data quality analysis
|   |-- data_cleaner.py             # Dynamic cleaning actions
|   |-- visualizations.py           # 6 Plotly chart functions
|   |-- predictive_model.py         # Logistic regression pipeline
|   |-- ai_narratives.py            # OpenAI GPT integration
|   |-- report_generator.py         # PDF report builder
|   |-- utils.py                    # Shared constants and helpers
```

---

## How to Run

```bash
# 1. Install dependencies (one-time)
pip install -r requirements.txt

# 2. Generate sample datasets (optional - they already exist)
python data/generate_dataset.py
python data/generate_health_data.py

# 3. Launch the dashboard
python -m streamlit run app.py

# 4. Open in browser
# http://localhost:8501
```

---

## The 7-Step Wizard Workflow

### Step 1: Upload Data

**What happens:**
- You are presented with two options:
  - **Option A:** Upload your own CSV or Excel file using the file uploader
  - **Option B:** Load the built-in sample education dataset (1,485 rows)
- After loading, the app shows a preview of the first 10 rows
- The data is stored in `st.session_state.raw_df` for use in all subsequent steps

**Key file:** `src/data_loader.py`
- `load_uploaded_file()` — reads CSV or Excel, returns DataFrame + error message
- `load_sample_data()` — loads `data/education_data.csv` from disk

**What you should look for:**
- Confirm the row/column count looks right
- Verify the preview shows your data correctly
- Check that column headers were parsed properly

---

### Step 2: Profile & Map Columns

**What happens:**
- The app **automatically analyzes every column** in your dataset and reports:
  - Data types (numeric, categorical, boolean, datetime)
  - Missing value counts and percentages
  - Outlier detection using the IQR method
  - Whitespace and casing inconsistencies
  - High cardinality warnings
  - Constant column detection
- Issues are displayed in a table with severity levels (red/yellow/green)
- You then **map your columns** to analytical roles:
  - **ID column** — unique identifier (e.g., student_id, patient_id)
  - **Target column** — the main variable to analyze/predict (e.g., grade, readmitted)
  - **Time column** — for trend analysis (e.g., semester, quarter)
  - **Numeric features** — continuous variables for charts and model inputs
  - **Categorical features** — grouping variables for breakdowns
- The app **auto-suggests** mappings based on column names and data types

**Key file:** `src/data_profiler.py`
- `profile_dataset()` — generates a full quality report for every column
- `suggest_column_mapping()` — intelligently guesses which columns serve which role
- `generate_cleaning_plan()` — creates an ordered list of recommended fixes

**Why this matters:**
- The column mapping drives everything downstream — visualizations, predictions, and AI analysis all use these mappings
- Getting this right ensures the dashboard generates meaningful insights for YOUR data, not just hardcoded education fields

---

### Step 3: Clean Data

**What happens:**
- The app presents all detected issues as a **checklist of cleaning actions**
- Each action has a checkbox, description, and impact preview
- Actions include:
  - **Remove duplicates** — drops identical rows
  - **Strip whitespace** — trims leading/trailing spaces from text columns
  - **Standardize casing** — converts text to Title Case for consistency
  - **Impute missing values** — fills NaN with median (numeric) or mode (categorical)
  - **Clip outliers** — constrains extreme values to IQR bounds
  - **Drop constant columns** — removes columns with only one unique value
  - **Convert to numeric** — coerces string numbers to numeric type
- You **check/uncheck** which fixes to apply (safe ones are pre-checked)
- After clicking "Apply", the app shows:
  - A log of every action taken
  - Before/after row counts
  - Remaining missing value count
  - Preview of the cleaned data

**Key file:** `src/data_cleaner.py`
- `apply_cleaning_actions()` — executes the selected actions, returns (cleaned_df, log)
- Each action is a dict like `{"action": "impute_missing", "column": "bmi", "method": "median"}`

**Important:** The original data is never modified. The cleaned copy is stored separately in `st.session_state.clean_df`.

---

### Step 4: Visualize

**What happens:**
- The app generates **up to 6 interactive Plotly charts**, all dynamically built from your column mappings:

1. **Target Distribution** — Histogram or bar chart of your target variable
   - Shows the shape of your data (normal? skewed? bimodal?)
   - Numeric targets get a histogram; categorical targets get a bar chart

2. **Correlation Scatter** — Scatter plot with OLS trendline
   - You choose X-axis, Y-axis, and optional color grouping via dropdowns
   - Displays Pearson correlation coefficient (r value) as an annotation
   - Bubble size can encode a third variable

3. **Distribution by Category** — Box plot with mean markers
   - Pick any numeric variable and any categorical variable
   - Shows median, quartiles, outliers, and diamond-shaped mean markers
   - Reveals how distributions differ across groups

4. **Category Breakdown** — Grouped bar chart
   - For binary targets: shows positive/negative rates by group with an overall reference line
   - For continuous targets: shows mean value per category
   - Helps identify which subgroups perform differently

5. **Trends Over Time** — Multi-line chart (only if a time column was mapped)
   - One line per category group (e.g., per department or course)
   - Dashed black line shows the overall average
   - Reveals improving/declining trends

6. **Correlation Heatmap** — Matrix of all numeric column correlations
   - Color-coded from -1 (red) to +1 (blue)
   - Quickly shows which variables are related

- A **filter panel** at the top lets you narrow data by any categorical or numeric range
- All charts are interactive: hover for details, click legend to toggle, zoom/pan

**Key file:** `src/visualizations.py`
- Each chart is a pure function that takes a DataFrame + column names and returns a Plotly Figure
- The app calls `st.plotly_chart()` to render them

---

### Step 5: Predict

**What happens:**
- A **logistic regression model** is trained on your data to predict the target variable
- You choose which numeric and categorical features to include
- The app displays:

  **Model Performance:**
  - Accuracy, Precision, Recall, F1 Score as metric cards
  - Confusion matrix as a color-coded heatmap
  - Feature importance as a horizontal bar chart (model coefficients)
  - Positive coefficients push toward the positive class; negative push away

  **Individual Predictor:**
  - A form with sliders and dropdowns for each feature
  - Enter any hypothetical values and click "Predict"
  - Shows the predicted class and confidence probability
  - A gauge chart visualizes the prediction confidence (0-100%)

**Key file:** `src/predictive_model.py`
- `train_model()` — builds a full sklearn pipeline:
  1. `StandardScaler` for numeric features
  2. `OneHotEncoder` for categorical features
  3. `LogisticRegression` classifier
  4. 80/20 train/test split with stratification
- For non-binary targets (e.g., grade 0-100), it auto-converts to binary (above/below median)
- `predict_single()` — predicts one record and returns (class, probability)

**Why logistic regression?**
- Fully interpretable — coefficients directly tell you "increasing X by 1 unit changes the prediction by Y"
- Fast to train on any dataset size
- Coefficients map naturally to actionable recommendations in the AI narrative

---

### Step 6: AI Insights

**What happens:**
- This step uses the **OpenAI GPT API** to generate human-readable narrative analysis
- **Requires:** An OpenAI API key entered in the sidebar (uses `gpt-4o-mini` for cost efficiency)
- Three types of AI-generated content:

  1. **Comprehensive Data Analysis** — A multi-paragraph narrative covering:
     - Population overview and key metrics
     - Correlations and patterns discovered
     - Equity/demographic comparisons
     - Time-based trends
     - Actionable recommendations

  2. **Model Interpretation** — Explains the predictive model in plain English:
     - How well it performs and what that means
     - Which features matter most and why
     - Practical recommendations based on feature importance

  3. **Per-Visualization Insights** — Short 2-3 sentence interpretations of specific charts

**Key file:** `src/ai_narratives.py`
- Statistics are computed **locally** and sent as structured JSON to GPT — raw data never leaves your machine
- Each API call costs approximately $0.001-0.005 (using gpt-4o-mini)
- Results are cached in session state to avoid redundant API calls
- All prompts use `temperature=0.3` for factual consistency

**Without an API key:** The dashboard still fully functions — you just won't see narrative text. All charts, filters, model, and exports work independently.

---

### Step 7: Report

**What happens:**
- Generate and download deliverables:

  1. **PDF Report** — A formatted document containing:
     - Title page with timestamp
     - Key statistics table (records, means, correlations, etc.)
     - Predictive model performance metrics
     - AI-generated narrative analysis (if API key was provided)
     - Can be generated with or without AI narratives

  2. **Cleaned Dataset (CSV)** — Export the cleaned, filtered data

  3. **Cleaning Log (TXT)** — Record of every cleaning action applied

  4. **Statistics (JSON)** — Machine-readable summary statistics

- A **workflow summary** shows everything that was accomplished across all 7 steps

**Key file:** `src/report_generator.py`
- Uses `fpdf2` library to create the PDF
- Handles Unicode by falling back to latin-1 encoding
- Returns `bytes` for direct use with Streamlit's download button

---

## Sample Datasets

### Education Dataset (`data/education_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| student_id | String | Unique student ID (STU-0001 to STU-0500) |
| gender | Categorical | Male, Female, Non-Binary |
| age | Integer | 17-25 |
| study_hours | Float | Weekly study hours (1-40), ~3% missing |
| attendance_rate | Float | Class attendance percentage (40-100%), ~3% missing |
| previous_grades | Float | Prior GPA (1.0-4.0), ~3% missing |
| parental_education | Categorical | No Degree, High School, Bachelor, Master, PhD |
| family_income_level | Categorical | Low, Medium, High |
| course | Categorical | Mathematics, Science, English, History, Computer Science |
| semester | Categorical | Fall 2023, Spring 2024, Fall 2024, Spring 2025 |
| grade | Float | Course grade (0-100) — correlated with study hours, attendance, GPA |
| passed | Binary | 1 if grade >= 50, else 0 |

**Built-in correlations:** Higher study hours and attendance drive higher grades. Parental education and family income have moderate positive effects. ~3% missing values injected for cleaning demo.

### Health Dataset (`data/health_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| patient_id | String | Unique patient ID (PAT-00001 to PAT-00600) |
| gender | Categorical | Male, Female |
| age | Integer | 1-95 (normal distribution around 55) |
| blood_type | Categorical | A+, A-, B+, B-, O+, O-, AB+, AB- |
| bmi | Float | Body mass index (15-50), ~3% missing |
| blood_pressure_systolic | Integer | Systolic BP (80-200), ~3% missing |
| heart_rate | Integer | Beats per minute (50-130), ~3% missing |
| department | Categorical | Cardiology, Orthopedics, Neurology, Oncology, Pediatrics |
| admission_type | Categorical | Emergency, Elective, Urgent |
| insurance_type | Categorical | Private, Medicare, Medicaid, Uninsured |
| quarter | Categorical | Q1-Q4 2024 |
| length_of_stay | Integer | Days in hospital (1-30) |
| treatment_cost | Float | Total cost in USD ($500-$80,000) |
| satisfaction_score | Float | Patient satisfaction (1-10), ~3% missing |
| readmitted | Binary | 1 if readmitted within 30 days, else 0 |
| discharge_status | Categorical | Recovered, Improved, Transferred, Deceased |

**Built-in correlations:** Older patients and emergency admissions have longer stays and higher readmission rates. Oncology/Cardiology departments have higher costs. Uninsured patients have higher readmission rates. Casing and whitespace inconsistencies injected for cleaning demo.

---

## Sidebar Features

The sidebar is always visible and contains:

1. **Progress Tracker** — Shows which of the 7 steps you've completed (checkmarks), are currently on (arrow), or haven't reached yet
2. **Navigation** — "Previous Step" button and quick-jump links to completed steps
3. **AI Settings** — OpenAI API key input (password field)
4. **Dataset Info** — Row count, column count, and cleaned row count (appears after upload)

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Streamlit** over Flask/Django | Fastest path from code to interactive dashboard. Widgets, layouts, and caching are built-in. |
| **Dynamic column mapping** over hardcoded schema | Allows the tool to work with ANY CSV dataset, not just education data |
| **Statistics-to-GPT** over raw-data-to-GPT | Keeps API costs near zero, avoids token limits, works with any dataset size |
| **gpt-4o-mini** over gpt-4o | Adequate for summarizing pre-computed statistics. 10-20x cheaper. |
| **Logistic Regression** over Random Forest/XGBoost | Fully interpretable coefficients that map to actionable recommendations |
| **fpdf2** over ReportLab | Simpler, lighter, sufficient for text-heavy reports |
| **Session state wizard** over multi-page app | Single-page with steps is simpler to deploy and gives a guided experience |

---

## Key Files Explained

### `app.py` (915 lines)
The main entry point. It is organized as a large if/elif chain — one block per wizard step. Each block:
- Reads data from `st.session_state`
- Renders the UI for that step
- Stores results back to `st.session_state`
- Provides a "Continue" button that advances to the next step

### `src/data_profiler.py` (286 lines)
The most analytically dense module. For each column it:
- Infers the semantic type (numeric, categorical, boolean, datetime)
- Computes statistics (mean, median, std, quartiles for numeric; top values for categorical)
- Detects issues (missing values, outliers, whitespace, casing, high cardinality, constants)
- Assigns severity levels (high/medium/low)
- Suggests fixes

### `src/visualizations.py` (197 lines)
Six pure functions, each returning a Plotly Figure. They accept column names as parameters so they work with any dataset. No Streamlit code inside — the caller handles rendering.

### `src/ai_narratives.py` (151 lines)
Four GPT prompt functions, each following the same pattern:
1. Accept pre-computed statistics (not raw data)
2. Build a system prompt defining the analyst role
3. Build a user prompt with the stats as JSON
4. Call `gpt-4o-mini` with low temperature
5. Return the response text

### `src/predictive_model.py` (124 lines)
A flexible sklearn pipeline builder that:
- Accepts any combination of numeric and categorical features
- Auto-handles binary vs continuous targets
- Returns the trained pipeline, metrics dict, and feature importances
- Includes a single-record prediction function for the interactive form
