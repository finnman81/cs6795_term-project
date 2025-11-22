#!/bin/bash
# Convenience script to run the Streamlit app

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Make sure OPENAI_API_KEY is set in your environment."
fi

# Run Streamlit app
streamlit run src/ui/1_Chatbot.py

