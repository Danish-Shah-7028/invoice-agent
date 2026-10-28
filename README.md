---
title: Invoice A2A Agent
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# A2A Invoice Agent (GA5 Q2)

Files: `app.py` (server), `requirements.txt`, `Dockerfile`, `test_smoke.py` (local test).

## 1. Deploy on Hugging Face Spaces (free, public HTTPS)

1. huggingface.co → **New Space** → SDK **Docker** → Blank → Public.
2. Upload `app.py`, `requirements.txt`, `Dockerfile`, and this `README.md` (the header block above is required).
3. **Settings → Variables and secrets**:

| Name | Type | Value |
|---|---|---|
| `LLM_API_KEY` | Secret | your AI Pipe token (aipipe.org, log in with your IITM email) |
| `LLM_BASE_URL` | Variable | `https://aipipe.org/openai/v1` |
| `LLM_MODEL` | Variable | `gpt-4.1-mini` (or `gpt-4.1-nano` to be cheaper) |
| `PUBLIC_BASE_URL` | Variable | `https://<username>-<spacename>.hf.space/a2a/` |
| `DEBUG_LOG_PACKAGES` | Variable | `1` (optional, prints the first package to the logs) |

Any OpenAI-compatible provider works: for Gemini use
`LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai` and `LLM_MODEL=gemini-2.5-flash`.

4. Wait for **Running**, then check:
```
curl https://<username>-<spacename>.hf.space/.well-known/agent-card.json
```
5. Paste `https://<username>-<spacename>.hf.space/a2a/` into the answer box. It must be **identical** to
   `PUBLIC_BASE_URL` (including the trailing slash).

## 2. Before you spend a Check

Run one Check, then open the Space **Logs**:
- The log line `first package:` shows the real package format. Confirm the evidence references look like
  `[SOMETHING]` inside the document text.
- If the feedback says the evidence refs are wrong because brackets are expected, set `REF_KEEP_BRACKETS=1`.
- `LLM call failed` lines mean the key, base URL, or model name is wrong. Fix them before the next Check:
  failed calls fall back to `open_exception` and score poorly.

## 3. Test locally (no API key needed)
```
pip install -r requirements.txt requests
MOCK_LLM=1 uvicorn app:app --port 7860
python test_smoke.py
```

## What the server does
- Agent Card is public; every other route needs `Authorization: Bearer <token>`. Each token is a separate user.
- Requires `A2A-Version: 1.0` (else 400) and `Content-Type: application/a2a+json` (else 415).
- First message → AI decides each package → Task in `TASK_STATE_INPUT_REQUIRED` with one proposals artifact.
- Results continuation → validated against the stored task/batch/package/actionId/action → `TASK_STATE_COMPLETED`
  with a receipts artifact listing only ACCEPTED actions. Cancel and results race safely: one gets 200, the other 409.
- Idempotency on (token, messageId); same message replays the same Task, changed content gives 409 `IDEMPOTENCY_CONFLICT`.
- Decisions are cached by policy revision + package content, so repeated packages never call the model again.
- Runs with a single worker on purpose: the locks that prevent double payments live in that one process.
