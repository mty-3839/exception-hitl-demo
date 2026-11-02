Live Demo: https://exception-hitl-demo-krgj83g3sefmthqxnfmv7s.streamlit.app/
# Exception HITL Demo

Lightweight AI product demo for job interviews: **Exception → classify + action + confidence → human-in-the-loop → decision log / eval**.

No database. No vector store. State lives in Streamlit `session_state`.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set the API key (pick one):

- Environment: `set OPENAI_API_KEY=sk-...`
- Or copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` and paste the key

```bash
streamlit run app.py
```

If you do not have a key yet, turn on **Offline mock** in the sidebar. The closed loop still works; only the model call is faked.

## Flow

1. Load a sample exception or paste your own.
2. **Analyze** — the model returns category, suggested action, confidence, uncertainty, and whether to escalate.
3. **HITL** — edit the action, force escalate, or override to auto-resolve.
4. **Confirm** — appends to the in-memory decision log.
5. Stats update: average confidence, escalate rate, human-edit rate.
