"""Streamlit page for post-study survey responses."""

import streamlit as st
from pathlib import Path
import sys
import csv
from datetime import datetime

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up from pages/ -> ui/ -> src/ -> project_root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

SURVEY_LOG_PATH = config.PROJECT_ROOT / "experiments" / "post_study_survey.csv"

# Survey CSV headers matching the actual survey questions
SURVEY_HEADERS = [
    "timestamp",
    "participant_name",
    "less_mentally_demanding",
    "preferred_layout",
    "preference_reason",
    "easier_to_understand",
    "easier_to_find_main_points",
    "more_overwhelming",
    "additional_feedback"
]


def init_survey_log_file(path: Path) -> None:
    """Initialize survey log file with correct headers if it doesn't exist or has wrong headers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and has correct headers
    if path.exists():
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_headers = reader.fieldnames
                # Check if headers match survey headers
                if existing_headers == SURVEY_HEADERS:
                    return  # File exists with correct headers
                else:
                    # File exists but has wrong headers - backup and recreate
                    backup_path = path.with_suffix('.csv.backup')
                    import shutil
                    shutil.copy2(path, backup_path)
                    # Recreate with correct headers
                    with open(path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=SURVEY_HEADERS)
                        writer.writeheader()
        except Exception:
            # If we can't read it, recreate it
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SURVEY_HEADERS)
                writer.writeheader()
    else:
        # Create file with survey headers
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SURVEY_HEADERS)
            writer.writeheader()


def save_survey_response(data):
    """Append survey responses to CSV."""
    # Always initialize to ensure correct headers
    init_survey_log_file(SURVEY_LOG_PATH)
    
    # Ensure all headers are present in data (fill missing with empty strings)
    complete_data = {header: data.get(header, "") for header in SURVEY_HEADERS}
    
    with open(SURVEY_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SURVEY_HEADERS)
        writer.writerow(complete_data)


def main():
    st.title("Post-Study Survey")
    st.write("Please answer the following questions about your experience with the chatbot layouts.")

    with st.form("post_study_survey"):
        participant_name = st.text_input("First Name")
        less_mentally_demanding = st.radio(
            "Which layout felt less mentally demanding overall?",
            ["Standard Layout", "Structured Layout", "About the same"]
        )
        preferred_layout = st.radio(
            "Which layout would you prefer to use again?",
            ["Standard Layout", "Structured Layout", "No preference"]
        )
        preference_reason = st.text_area(
            "In a few words, why did you prefer that layout?",
            help="Optional but helpful for our research."
        )
        easier_to_understand = st.radio(
            "Which layout made it easier to understand the information?",
            ["Standard Layout", "Structured Layout", "About the same"]
        )
        easier_to_find_main_points = st.radio(
            "Which layout made it easier to find the main points or strategies?",
            ["Standard Layout", "Structured Layout", "About the same"]
        )
        more_overwhelming = st.radio(
            "Which layout felt more overwhelming or cluttered?",
            ["Standard Layout", "Structured Layout", "About the same"]
        )
        additional_feedback = st.text_area(
            "Is there anything else you’d like to share about your experience using either layout?"
        )

        submitted = st.form_submit_button("Submit")
        if submitted:
            if not participant_name.strip():
                st.warning("Please enter your first name before submitting.")
            else:
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "participant_name": participant_name.strip(),
                    "less_mentally_demanding": less_mentally_demanding,
                    "preferred_layout": preferred_layout,
                    "preference_reason": preference_reason.strip(),
                    "easier_to_understand": easier_to_understand,
                    "easier_to_find_main_points": easier_to_find_main_points,
                    "more_overwhelming": more_overwhelming,
                    "additional_feedback": additional_feedback.strip()
                }
                save_survey_response(response)
                st.success("Thank you! Your responses have been recorded.")


if __name__ == "__main__":
    main()

