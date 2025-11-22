# Deployment Guide for Streamlit Community Cloud

This guide will help you deploy the Cognitive-Load-Aware Chatbot to Streamlit Community Cloud.

## Prerequisites

1. GitHub account
2. Streamlit Community Cloud account (free at https://share.streamlit.io)
3. OpenAI API key

## Step 1: Prepare Your Repository

1. **Ensure all necessary files are committed:**
   - Vector store files (`data/vector_store.faiss` and `data/vector_store.faiss.metadata.json`)
   - Processed documents (`data/processed_docs.jsonl`)
   - All source code
   - `requirements.txt`
   - `.streamlit/config.toml`

2. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

## Step 2: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `cs-6795`
5. Set the main file path: `src/ui/1_Chatbot.py`
6. Click "Deploy!"

## Step 3: Configure Secrets

1. In your Streamlit Cloud app dashboard, go to **Settings** → **Secrets**
2. Add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
3. Click "Save"

## Step 4: Verify Deployment

1. Wait for the app to deploy (usually 1-2 minutes)
2. Visit your app URL (e.g., `https://your-app-name.streamlit.app`)
3. Test the chatbot to ensure it works
4. Check that results are being saved (they'll be in the app's filesystem)

## Accessing Results

Results are saved to CSV files in the `experiments/` directory:
- `experiments/results_raw.csv` - Chatbot interactions
- `experiments/post_study_survey.csv` - Survey responses

**To download results:**
- Option 1: Add a simple admin page to view/download results (recommended)
- Option 2: Access via Streamlit Cloud's file browser (if available)
- Option 3: Use git to pull the files (if you commit them)

## Troubleshooting

- **"ModuleNotFoundError"**: Ensure all dependencies are in `requirements.txt`
- **"OPENAI_API_KEY not found"**: Check that secrets are configured correctly
- **"Vector store not found"**: Ensure vector store files are committed to git
- **App won't start**: Check the logs in Streamlit Cloud dashboard

## Notes

- The app is public by default (anyone with the URL can access it)
- Results are stored in the app's filesystem and persist between restarts
- For production use, consider adding authentication or password protection

