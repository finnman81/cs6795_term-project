@echo off
REM Convenience script to run the Streamlit app on Windows

REM Check if .env file exists
if not exist .env (
    echo Warning: .env file not found. Make sure OPENAI_API_KEY is set in your environment.
)

REM Run Streamlit app
streamlit run src/ui/1_Chatbot.py

