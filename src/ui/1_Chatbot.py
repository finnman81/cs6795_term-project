"""Main Streamlit application entry point."""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

# Ensure project root is on sys.path when launched via `streamlit run src/ui/1_Chatbot.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.embeddings import load_vector_store
from src.rag_pipeline import answer_query
from src.logging_utils import log_interaction, log_error
from src.ui.layout_baseline import render_baseline_view
from src.ui.layout_cl_aware import render_cl_aware_view


# Page configuration
st.set_page_config(
    page_title="Cognitive-Load-Aware RAG",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "tasks" not in st.session_state:
    st.session_state.tasks = None


@st.cache_resource
def load_vector_store_cached():
    """Load vector store with caching."""
    try:
        return load_vector_store(config)
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        st.stop()


def load_tasks() -> Optional[list]:
    """Load tasks from tasks.json."""
    tasks_path = config.PROJECT_ROOT / "experiments" / "tasks.json"
    if tasks_path.exists():
        with open(tasks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Load vector store
try:
    st.session_state.vector_store = load_vector_store_cached()
except Exception as e:
    st.error(f"Failed to load vector store: {str(e)}")
    st.info("Please run `python scripts/build_index.py` first to build the vector index.")
    st.stop()

# Load tasks
st.session_state.tasks = load_tasks()

# Main UI
st.title("🧠 Cognitive-Load-Aware Chatbot")
st.markdown(
    """
    **About this project**
    
    This tool is a research prototype that helps you find practical strategies for digital wellbeing and focus. It searches a small collection of articles about managing distractions, phone use, multitasking, and attention, then uses an AI model to summarize the most relevant ideas for your question. We’re studying how different ways of presenting these answers can make it easier to understand and act on the advice without feeling overwhelmed. Please use it for general guidance only. It is not a substitute for professional medical or mental health care.

    **Instructions**
    
    You’ll use this tool to answer a few brief tasks (for example, finding strategies to reduce distractions), then tell us how clear and usable the answers felt. We’re comparing different layouts to learn which one makes it easier to understand and apply the advice.
    """
)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    participant_id = st.text_input(
        "First Name",
        value="",
        help="Enter participant’s first name for experiment logging"
    )
    
    response_type = st.selectbox(
        "Response Type",
        options=["Standard Layout", "Structured Layout"],
        help="Select interface presentation style"
    )
    condition = "baseline" if response_type == "Standard Layout" else "cl_aware"
    
    # Task selection
    use_task = st.checkbox("Use predefined task", value=False)
    
    selected_task = None
    if use_task and st.session_state.tasks:
        task_options = {f"Task {t['task_id']}: {t['prompt'][:50]}...": t for t in st.session_state.tasks}
        selected_task_key = st.selectbox("Select Task", options=list(task_options.keys()))
        selected_task = task_options[selected_task_key]
    elif use_task and not st.session_state.tasks:
        st.warning("tasks.json not found. Using free-form query mode.")

# Main content area
if use_task and selected_task:
    st.header(f"Task {selected_task['task_id']}")
    st.write(selected_task['prompt'])
    query = selected_task['prompt']
    task_id = selected_task['task_id']
else:
    st.header("Questions")
    query = st.text_area(
        "Enter your question about digital wellbeing and focus:",
        height=100,
        placeholder="e.g., How can I reduce phone use while working?"
    )
    
    st.caption("Need inspiration? Try one of these:")
    sample_questions = [
        "How can I build a morning routine that reduces phone dependence?",
        "What are practical steps to cut back on doomscrolling at night?",
        "How do I stay focused when constant notifications pull me away?",
        "What strategies help balance work tasks without multitasking overload?",
        "How can I set boundaries with social media while still staying informed?"
    ]
    st.markdown("\n".join([f"- {q}" for q in sample_questions]))
    task_id = None

# Submit button
if st.button("Submit Query", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    elif not participant_id.strip():
        st.warning("Please enter a first name for logging.")
    else:
        start_time = datetime.now()
        
        with st.spinner("Processing query..."):
            try:
                # Get answer from RAG pipeline
                result = answer_query(
                    query=query,
                    condition=condition,
                    vector_store=st.session_state.vector_store,
                    config=config
                )
                
                end_time = datetime.now()
                
                # Render based on condition
                if condition == "baseline":
                    render_baseline_view(
                        query=result["query"],
                        answer=result["answer"],
                        retrieved_chunks=result["retrieved_chunks"]
                    )
                else:  # cl_aware
                    render_cl_aware_view(
                        query=result["query"],
                        answer=result["answer"],
                        retrieved_chunks=result["retrieved_chunks"]
                    )
                
                # Log interaction
                try:
                    log_interaction(
                        participant_id=participant_id,
                        condition=condition,
                        task_id=task_id,
                        query=query,
                        answer=result["answer"],
                        retrieved_chunks=result["retrieved_chunks"],
                        start_time=start_time,
                        end_time=end_time,
                        log_path=config.LOG_FILE_PATH
                    )
                    st.success("Interaction logged successfully.")
                except Exception as e:
                    log_error(f"Failed to log interaction: {str(e)}")
                    st.error(f"Failed to log interaction: {str(e)}")
            
            except Exception as e:
                log_error(f"Error processing query: {str(e)}")
                st.error(f"Error processing query: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.caption("Cognitive-Load-Aware RAG System for Digital Wellbeing & Focus")

