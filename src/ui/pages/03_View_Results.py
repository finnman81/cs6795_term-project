"""Admin page to view and download experiment results."""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

RESULTS_RAW_PATH = config.PROJECT_ROOT / "experiments" / "results_raw.csv"
SURVEY_PATH = config.PROJECT_ROOT / "experiments" / "post_study_survey.csv"


def load_csv_if_exists(path: Path):
    """Load CSV file if it exists, return None otherwise."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Error reading {path.name}: {str(e)}")
            return None
    return None


def main():
    st.title("📊 Experiment Results")
    st.write("View and download participant data from the experiment.")
    
    # Chatbot Interactions
    st.header("Chatbot Interactions")
    results_df = load_csv_if_exists(RESULTS_RAW_PATH)
    
    if results_df is not None and not results_df.empty:
        st.write(f"**Total interactions:** {len(results_df)}")
        
        # Display summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Participants", results_df['participant_id'].nunique())
        with col2:
            baseline_count = len(results_df[results_df['condition'] == 'baseline'])
            st.metric("Standard Layout", baseline_count)
        with col3:
            cl_aware_count = len(results_df[results_df['condition'] == 'cl_aware'])
            st.metric("Structured Layout", cl_aware_count)
        
        # Display data
        st.subheader("Data Preview")
        st.dataframe(results_df, use_container_width=True, height=400)
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Chatbot Interactions (CSV)",
            data=csv,
            file_name=f"results_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No chatbot interaction data yet. Results will appear here as participants use the app.")
    
    st.markdown("---")
    
    # Survey Responses
    st.header("Post-Study Survey Responses")
    survey_df = load_csv_if_exists(SURVEY_PATH)
    
    if survey_df is not None and not survey_df.empty:
        st.write(f"**Total survey responses:** {len(survey_df)}")
        
        # Display summary stats (with safe column access)
        col1, col2 = st.columns(2)
        with col1:
            if 'participant_name' in survey_df.columns:
                st.metric("Survey Respondents", survey_df['participant_name'].nunique())
            else:
                st.metric("Survey Respondents", len(survey_df))
        with col2:
            if 'preferred_layout' in survey_df.columns:
                preferred_standard = len(survey_df[survey_df['preferred_layout'] == 'Standard Layout'])
                preferred_structured = len(survey_df[survey_df['preferred_layout'] == 'Structured Layout'])
                st.metric("Preferred Structured", f"{preferred_structured}/{len(survey_df)}")
            else:
                st.metric("Total Responses", len(survey_df))
        
        # Display data
        st.subheader("Survey Data Preview")
        st.dataframe(survey_df, use_container_width=True, height=400)
        
        # Download button
        csv = survey_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Survey Responses (CSV)",
            data=csv,
            file_name=f"post_study_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Quick analysis (with safe column access)
        st.subheader("Quick Analysis")
        if 'less_mentally_demanding' in survey_df.columns:
            st.write("**Which layout felt less mentally demanding?**")
            st.bar_chart(survey_df['less_mentally_demanding'].value_counts())
        
        if 'preferred_layout' in survey_df.columns:
            st.write("**Which layout would participants prefer to use again?**")
            st.bar_chart(survey_df['preferred_layout'].value_counts())
        
        if 'easier_to_understand' in survey_df.columns:
            st.write("**Which layout made it easier to understand the information?**")
            st.bar_chart(survey_df['easier_to_understand'].value_counts())
    else:
        st.info("No survey responses yet. Survey data will appear here as participants complete the post-study survey.")


if __name__ == "__main__":
    main()

