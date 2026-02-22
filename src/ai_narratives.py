"""OpenAI API integration for generating AI-powered narrative insights."""

import json
import streamlit as st
from openai import OpenAI


def get_openai_client(api_key=None):
    """Initialize OpenAI client. Returns None if no key available."""
    if api_key:
        return OpenAI(api_key=api_key)
    return None


def generate_data_summary(client, stats):
    """Generate a comprehensive narrative summary from dataset statistics."""
    system_prompt = """You are an education data analyst. Given statistical summaries
of student performance data, write clear, professional narrative insights.
Use specific numbers from the data. Organize into paragraphs:
1. Overview of the student population and performance
2. Key correlations and patterns
3. Equity considerations across demographic groups
4. Semester-over-semester trends
5. Actionable recommendations for educators
Keep the tone professional but accessible. Use bullet points for recommendations."""

    user_prompt = f"""Analyze the following education dataset statistics and provide
a comprehensive narrative summary:

{json.dumps(stats, indent=2)}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def generate_visualization_insight(client, chart_name, stats):
    """Generate a 2-3 sentence interpretation of a specific chart."""
    system_prompt = """You are an education data analyst. Given statistics about a specific
visualization, write 2-3 concise sentences interpreting the key findings.
Be specific with numbers. Focus on actionable insights."""

    user_prompt = f"""Interpret the following data for the '{chart_name}' visualization:

{json.dumps(stats, indent=2)}

Provide a brief, insightful interpretation."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insight: {str(e)}"


def generate_predictive_insight(client, model_metrics, feature_importances):
    """Narrate the predictive model results."""
    system_prompt = """You are an education data analyst. Given machine learning model
performance metrics and feature importances, write a clear interpretation. Explain:
1. How well the model performs
2. Which factors most influence student success
3. Practical recommendations based on the findings
Keep it concise (3-4 paragraphs)."""

    data = {
        "model_type": "Logistic Regression",
        "target": "Student Pass/Fail Prediction",
        "metrics": model_metrics,
        "feature_importances": feature_importances,
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Interpret these model results:\n\n{json.dumps(data, indent=2)}"},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insight: {str(e)}"


def generate_full_report(client, stats, model_metrics, feature_importances):
    """Generate a complete analytical report combining all insights."""
    system_prompt = """You are a senior education data analyst writing a formal analytical report.
Given comprehensive dataset statistics and predictive model results, produce a structured report with:

# Executive Summary
Brief overview of findings (1 paragraph)

# Student Population & Performance Overview
Key demographics and grade distribution analysis

# Key Correlations & Patterns
Analysis of factors affecting student performance

# Equity Analysis
Examination of performance across demographic groups

# Predictive Model Findings
Interpretation of the ML model results and key predictors

# Semester Trends
Analysis of how performance has changed over time

# Recommendations
5-7 specific, actionable recommendations for educators and administrators

Use specific numbers from the data. Be professional and thorough."""

    data = {
        "dataset_statistics": stats,
        "model_metrics": model_metrics,
        "feature_importances": feature_importances,
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a full analytical report from this data:\n\n{json.dumps(data, indent=2)}"},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"
