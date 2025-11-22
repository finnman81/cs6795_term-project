# Pre-Deployment Checklist

Before deploying to Streamlit Cloud, verify the following:

## ✅ Files to Commit

- [ ] `src/` - All source code
- [ ] `data/vector_store.faiss` - Vector index (required)
- [ ] `data/vector_store.faiss.metadata.json` - Vector metadata (required)
- [ ] `data/processed_docs.jsonl` - Processed documents (required)
- [ ] `experiments/tasks.json` - Task definitions (if using)
- [ ] `requirements.txt` - Python dependencies
- [ ] `.streamlit/config.toml` - Streamlit configuration
- [ ] `DEPLOYMENT.md` - This guide

## ✅ Files to NOT Commit (already in .gitignore)

- [x] `.env` - Contains API keys (use Streamlit secrets instead)
- [x] `data/*.pdf` - PDF files are too large
- [x] `experiments/*.csv` - Result files (will be generated)
- [x] `__pycache__/` - Python cache files

## ✅ Configuration

- [ ] Verify `requirements.txt` has all dependencies
- [ ] Test locally that the app runs: `streamlit run src/ui/1_Chatbot.py`
- [ ] Ensure vector store files exist and are accessible

## ✅ Streamlit Cloud Setup

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set main file path: `src/ui/1_Chatbot.py`
4. Add secret: `OPENAI_API_KEY` in Streamlit Cloud settings

## Quick Test Commands

```bash
# Test locally
streamlit run src/ui/1_Chatbot.py

# Check what will be committed
git status

# Verify vector store exists
ls -lh data/vector_store.faiss*
```

