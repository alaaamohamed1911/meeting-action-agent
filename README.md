# Meeting → Action AI Agent (Streamlit)

Adapted **from `Meeting_Action_Agent_(1).ipynb`**. Preserves Gemini structured extraction, meeting summary / decisions / suggestions, date and evidence validation, LangGraph interrupt/resume human approval, SQLite task persistence, deduplication, prompt-injection heuristic checks, checkpointing, and saved-task function calling.

## Deploy on Streamlit Community Cloud

1. Upload this directory (its **contents**, including `app.py`, `backend/`, and `requirements.txt`) to a **private GitHub repository**.
2. Go to https://share.streamlit.io/ and create an app from the repo with **main file: `app.py`**.
3. Set Secrets (TOML):
   ```toml
   GEMINI_API_KEY = "your-real-key"
   GEMINI_MODEL = "gemini-3.1-flash-lite"
   ```
   Never commit your real key; `.streamlit/secrets.toml` is gitignored.
4. Deploy, enter a meeting, select task decisions, and click **Submit Review & Save Approved Tasks**.

## Local run

Python 3.10+ recommended. From this folder:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"      # PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

Tests: `python -m pytest -q`.

## Important safety and hosting limitations

- **Proof of concept only.** No user accounts/permissions: everyone who can access the app could see every saved task and review thread. Use only fake/test meeting data until authentication, tenancy, and authorization exist.
- **Streamlit Community Cloud filesystem is not durable.** Both `meeting_agent.db` and `meeting_checkpoints.db` live under `data/` and may be lost on restart/redeploy. For durable or multi-user production use, replace local SQLite with a persistent supported database/checkpoint store.
- The original notebook's approval policy is preserved: `blocked` tasks cannot be approved; `needs_review` tasks can be explicitly approved (for example `event_based` deadlines), and save validation checks again. The UI currently does **not** edit task fields.
- Regex security checks only flag certain phrases; they do not guarantee immunity to prompt injection.
- The `get_saved_tasks` tool is allowlisted but the demo has no per-user data isolation.
- No Colab interaction (`input()`, `%pip`, `google.colab.userdata`) is required.
- For model availability, `GEMINI_MODEL` can be changed in Secrets.

## Files

- `app.py`: Streamlit meeting input, review, results, saved tasks, and saved-task Q&A.
- `backend/agent.py`: extracted and adapted notebook backend (one LangGraph graph).
- `tests/test_core.py`: selected offline regression tests.
- `requirements.txt`: dependencies.
