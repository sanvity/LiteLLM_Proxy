# 🛡️ LiteLLM PII-Shield Proxy — Complete User Guide

> **Who is this for?** Anyone running this project for the first time. No prior knowledge of Python environments, LLMs, or terminal commands is assumed. Every step is explained end-to-end.

---

## 📌 Table of Contents

1. [What Does This Project Do?](#1-what-does-this-project-do)
2. [System Architecture](#2-system-architecture)
3. [Project File Structure](#3-project-file-structure)
4. [Prerequisites](#4-prerequisites)
5. [Installation](#5-installation)
6. [Environment Configuration](#6-environment-configuration)
7. [Running the Application](#7-running-the-application)
8. [Using the Interfaces](#8-using-the-interfaces)
9. [Training and Fine-Tuning the PII Model](#9-training-and-fine-tuning-the-pii-model)
10. [Generating Evaluation Reports](#10-generating-evaluation-reports)
11. [Running the Test Suite](#11-running-the-test-suite)
12. [API Reference](#12-api-reference)
13. [Configuration Reference](#13-configuration-reference)
14. [Troubleshooting FAQ](#14-troubleshooting-faq)
15. [Glossary](#15-glossary)

---

## 1. What Does This Project Do?

This is an **enterprise-grade AI gateway** that sits between your application and multiple Large Language Model (LLM) providers — Groq, Cerebras, Together AI, and local Ollama. It provides three major capabilities out of the box:

| Capability | Plain-English meaning |
|---|---|
| **Intelligent Routing** | Automatically picks the cheapest or fastest AI model based on how complex your question is |
| **Load Balancing** | Spreads requests across multiple AI providers so no single one gets overloaded |
| **PII Guardrail** | Scans every message for Personally Identifiable Information (names, emails, IDs, etc.) and either blocks or masks it before it reaches any external AI |

**Example:** You send _"My email is alice@example.com — what is 2+2?"_  
The system replaces `alice@example.com` with `[EMAIL]` before forwarding to the AI, protecting user privacy automatically.

---

## 2. System Architecture

```
Your App / Browser
       |
       |  POST /v1/chat/completions
       v
FastAPI Proxy  (port 8000)
       |
       v
PII Guardrail  — Pre-Call Scan
  |-- BLOCK policy  -->  Return 400 Bad Request immediately
  |-- MASK policy   -->  Replace PII tokens e.g. [EMAIL], [PERSON]
  `-- No PII / IGNORE -->  Pass through unchanged
       |
       v
Complexity Classifier
  |-- LOW    -->  Cerebras Llama 8B   ($0.01 / 1M tokens)
  |-- MEDIUM -->  Groq Llama 8B       ($0.05 / 1M tokens)
  `-- HIGH   -->  Groq / Together 70B ($0.70 / 1M tokens)
       |
       v
LLM Response
       |
       v
PII Guardrail  — Post-Call Scan (masks any PII in the reply too)
       |
       v
Clean Response back to your app
```

**Two failover clusters:**
- **`primary-cluster`** — Groq + Cerebras (low latency, low cost)
- **`backup-cluster`** — Together AI + Groq (automatic failover if primary is down)

---

## 3. Project File Structure

```
LiteLLM Deployed/
|
|-- main.py                      Entry point — starts the FastAPI backend server
|-- app.py                       Streamlit admin console (browser UI on port 8501)
|-- config.yaml                  LLM clusters, routing rules, and fallback chains
|-- requirements.txt             All Python package dependencies
|-- .env                         YOUR secret API keys (you must create this file)
|-- .env.example                 Template — shows exactly which keys are needed
|-- synthetic_data.py            Synthetic PII data generation engine
|
|-- proxy/
|   |-- app.py                   FastAPI route controllers
|   |-- router.py                Complexity classifier + load balancer
|   |-- config.py                YAML config parser (Pydantic models)
|   `-- mlops.py                 Model version registry API
|
|-- guardrails/
|   |-- deberta_pii_guardrail.py DeBERTa-v3 PII detection (runs fully locally)
|   `-- safety_guardrails.py     Content safety rules engine
|
|-- models/
|   |-- finetuned-deberta/       Your fine-tuned PII model weights (git-ignored)
|   |-- versions/                Saved checkpoint snapshots
|   `-- versions_registry.json   Tracks all trained model versions + metrics
|
|-- frontend/
|   `-- index.html               Standalone browser dashboard (no server needed)
|
|-- scripts/
|   |-- evaluate.py              Run evaluation and generate HTML report
|   |-- train_pii_deberta.py     Fine-tune the DeBERTa PII model on new data
|   |-- run_synthetic_engine.py  Generate synthetic PII training data
|   |-- generate_test_set.py     Create a held-out test dataset (small)
|   `-- generate_scaled_test_set.py  Create a larger test dataset
|
|-- data/
|   |-- synthetic_dataset.json   Generated training data
|   |-- test_dataset_heldout.json  Held-out evaluation set
|   |-- evaluation_report.html   Latest HTML evaluation report (regenerate any time)
|   `-- evaluation_data.json     Cached metrics consumed by the live dashboard
|
|-- config/
|   |-- pii_guardrail_config.json    PII entity labels and thresholds
|   `-- safety_guardrail_config.json Content safety keyword rules
|
`-- tests/
    |-- test_proxy.py            Integration tests for routing + PII guardrail (15 tests)
    `-- test_synthetic_engine.py Tests for the synthetic data pipeline
```

---

## 4. Prerequisites

### 4.1 Python 3.11

Check your version:
```bash
python3 --version
```

You need **Python 3.11.x**. Python 3.12+ is **not supported** because of PyTorch compatibility.

Download Python 3.11 from: https://www.python.org/downloads/

### 4.2 uv — Fast Package Manager

`uv` installs dependencies 10-100x faster than pip.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal, then verify:
```bash
uv --version
```

---

## 5. Installation

### Step 1 — Navigate to the project folder

```bash
cd "/path/to/LiteLLM Deployed"
```

> macOS tip: type `cd ` then drag the folder onto the Terminal window to auto-fill the path.

### Step 2 — Create a virtual environment

```bash
uv venv --python 3.11
```

Expected output:
```
Using CPython 3.11.x
Creating virtual environment at: .venv
```

This creates an isolated `.venv/` folder so nothing touches your system Python.

### Step 3 — Install all dependencies

Run both commands in order:
```bash
uv pip install -r requirements.txt --index-strategy unsafe-best-match --python .venv/bin/python
```
```bash
uv pip install exrex --python .venv/bin/python
```

> This downloads ~2-3 GB (PyTorch + Transformers). It only runs once and takes 5-15 minutes depending on your internet speed.

**Key packages explained:**

| Package | What it does |
|---|---|
| `litellm` | Routes to Groq, Cerebras, Together AI, Ollama through one interface |
| `fastapi` + `uvicorn` | The API server that receives requests on port 8000 |
| `streamlit` | The admin console web UI on port 8501 |
| `transformers` + `torch` | Powers the local DeBERTa-v3 PII detection model |
| `peft` | Fine-tuning the PII model efficiently |
| `scikit-learn` + `plotly` + `pandas` | Evaluation metrics and interactive charts |
| `faker` + `exrex` | Generates realistic fake PII data for model training |

---

## 6. Environment Configuration

The app reads secrets from a `.env` file in the project root. You must create this before running.

### 6.1 Create the .env file

```bash
# macOS / Linux
cp .env.example .env
```

Then open `.env` in any text editor and fill in your keys:

```env
# ── Server ───────────────────────────────────────────
PORT=8000
HOST=0.0.0.0

# ── LLM Provider API Keys ────────────────────────────
# Groq  →  https://console.groq.com  →  API Keys  →  Create Key
GROQ_API_KEY=your_groq_key_here

# Cerebras  →  https://cloud.cerebras.ai  →  API Keys
CEREBRAS_API_KEY=your_cerebras_key_here

# Together AI  →  https://api.together.ai  →  Settings  →  API Keys
TOGETHERAI_API_KEY=your_togetherai_key_here

# ── Local Ollama (optional) ──────────────────────────
# Only needed if you want to run models locally with Ollama
OLLAMA_API_BASE=http://localhost:11434

# ── Internal URL ─────────────────────────────────────
PROXY_URL=http://localhost:8000
```

### 6.2 Where to get API keys — all free tiers available

| Provider | Free tier | Sign-up URL |
|---|---|---|
| **Groq** | Yes — fast, generous limits | https://console.groq.com |
| **Cerebras** | Yes — ultra-fast inference | https://cloud.cerebras.ai |
| **Together AI** | Yes — $1 free credit | https://api.together.ai |

> **Never commit your `.env` file.** It is already excluded by `.gitignore`.

### 6.3 No keys? Use mock mode

All 15 integration tests and most local features work without real API keys. The system has a built-in mock LLM backend for offline testing.

---

## 7. Running the Application

You need **two terminals open at the same time**.

### Terminal 1 — FastAPI Backend (port 8000)

```bash
cd "/path/to/LiteLLM Deployed"
.venv/bin/python main.py
```

Wait for this line before doing anything else:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Quick health check:
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

### Terminal 2 — Streamlit Admin Console (port 8501)

Open a **new** terminal window:
```bash
cd "/path/to/LiteLLM Deployed"
.venv/bin/streamlit run app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

### Option 3 — Static Frontend Dashboard (no server needed)

To view the routing dashboard without Streamlit, just open this file in any browser:
```
frontend/index.html
```

On macOS:
```bash
open "frontend/index.html"
```

The dashboard connects directly to the FastAPI backend on port 8000 — so the backend must still be running.

---

## 8. Using the Interfaces

### 8.1 Static Frontend (`frontend/index.html`)

**Gateway Routing tab:**
- **Chat Sandbox** — type a message, hit Send. The system picks the right model automatically.
- **Settings panel** — choose cluster (`primary-cluster` / `backup-cluster`) and PII policy (`IGNORE` / `MASK` / `BLOCK`).
- **Live Stats** — requests processed, cost spent, active providers, avg latency.
- **System Logs** — real-time streaming log from the routing engine.
- **Provider Catalog** — all configured LLM nodes with cost tiers and status.

**Guardrail Evaluation tab:**
- Shows cached F1 scores, confusion matrix, accuracy, and per-entity breakdowns comparing base vs. fine-tuned model.
- Data is refreshed by running `scripts/evaluate.py`.

### 8.2 Streamlit Console (`app.py` on port 8501)

Four pages accessible via tabs:

| Tab | What you can do |
|---|---|
| **Chat Testing** | Send messages through the proxy with live PII masking shown |
| **Model Training** | Launch fine-tuning runs on synthetic or custom data |
| **Training Data** | Generate and inspect synthetic PII datasets |
| **Model Registry / MLOps** | View saved model versions, deploy or delete them, trigger evaluation |

### 8.3 Direct API — OpenAI-compatible

The proxy is a drop-in OpenAI replacement. Use it with any OpenAI-compatible client:

**Python (openai SDK):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-checked-locally"
)

response = client.chat.completions.create(
    model="primary-cluster",
    messages=[{"role": "user", "content": "Summarise the theory of relativity."}]
)
print(response.choices[0].message.content)
```

**curl:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "primary-cluster",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "pii_policy": "MASK"
  }'
```

---

## 9. Training and Fine-Tuning the PII Model

The system ships with a pre-trained DeBERTa-v3 model in `models/finetuned-deberta/`. Fine-tuning is **optional** — the system works as-is. Only follow these steps to improve PII accuracy on your own data.

### Step 1 — Generate synthetic training data

```bash
.venv/bin/python scripts/run_synthetic_engine.py
```

For a larger dataset (better results, slower):
```bash
.venv/bin/python scripts/generate_scaled_test_set.py
```

Output is saved to `data/synthetic_dataset.json`.

### Step 2 — Train the model

```bash
.venv/bin/python scripts/train_pii_deberta.py
```

Default training parameters:
- Epochs: 15
- Learning rate: 0.0001
- Batch size: 8

Training runs on CPU and takes roughly 30-60 minutes. No GPU required.

The new model is saved to `models/finetuned-deberta/` and registered in `models/versions_registry.json`.

### Step 3 — Activate the new version

Via the Streamlit console → **Model Registry** tab → click **Deploy** next to your new version.

Or edit `models/versions_registry.json` directly and set `"active_version"` to your version ID.

---

## 10. Generating Evaluation Reports

```bash
.venv/bin/python scripts/evaluate.py
```

This will:
1. Load both the base Hugging Face model and your fine-tuned model
2. Run inference on the 84-sample held-out test set
3. Compute per-entity Precision, Recall, F1, and overall character-level Accuracy
4. Generate a confusion matrix and error analysis (fixed misses / new errors)
5. Save two output files:
   - `data/evaluation_report.html` — standalone visual report, open in any browser
   - `data/evaluation_data.json` — feeds the live dashboard automatically

Open the report:
```bash
open data/evaluation_report.html     # macOS
# or double-click the file in Finder
```

> The first run downloads the base DeBERTa-v3 model from Hugging Face (~400 MB). You'll see  
> _"You are sending unauthenticated requests to the HF Hub"_ — this is normal and harmless.

---

## 11. Running the Test Suite

```bash
.venv/bin/python -m unittest discover -s tests
```

Expected output:
```
Ran 15 tests in ~40s

OK
```

Run a specific file:
```bash
# Proxy routing + PII guardrail tests (11 tests)
.venv/bin/python -m unittest tests/test_proxy.py

# Synthetic data pipeline tests (4 tests)
.venv/bin/python -m unittest tests/test_synthetic_engine.py
```

---

## 12. API Reference

Base URL: `http://localhost:8000`

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status":"ok"}` if running |
| GET | `/metrics` | Prometheus metrics — requests, latency, cost per provider |
| POST | `/v1/chat/completions` | Main OpenAI-compatible chat endpoint |
| GET | `/api/evaluation` | Latest evaluation JSON for the frontend dashboard |
| GET | `/ui/mlops/registry` | All saved model versions |
| POST | `/ui/mlops/deploy` | Activate a model version |
| POST | `/ui/mlops/evaluate` | Trigger an evaluation run |

### POST /v1/chat/completions — request body

```json
{
  "model": "primary-cluster",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Your question here"}
  ],
  "max_tokens": 1000,
  "temperature": 0.7,
  "pii_policy": "MASK"
}
```

**`model` options:**
- `"primary-cluster"` — Groq + Cerebras, complexity-aware routing (recommended)
- `"backup-cluster"` — Together AI + Groq, used automatically on failover

**`pii_policy` options:**
- `"IGNORE"` — no scanning, fastest
- `"MASK"` — replace PII with `[ENTITY_TYPE]` placeholders (default)
- `"BLOCK"` — reject the entire request with HTTP 400 if any PII is detected

---

## 13. Configuration Reference

`config.yaml` controls the full routing topology.

```yaml
model_list:
  - model_name: primary-cluster        # Name used in API "model" field
    litellm_params:
      model: groq/llama-3.1-8b-instant # provider/model-id
      api_key: os.environ/GROQ_API_KEY  # reads from .env automatically
      rpm: 30                           # max Requests Per Minute
      tpm: 100000                       # max Tokens Per Minute
      tpr: 131072                       # max Tokens Per Request (context window)
      cost_per_million: 0.05            # USD per 1M tokens — used for routing decisions
      complexity_tier: medium           # low / medium / high

router_settings:
  routing_strategy: usage-based-routing # distributes load by current TPM usage
  allowed_fails: 1                      # failures before switching nodes
  cooldown_time: 60                     # seconds before retrying a failed node
  num_retries: 3                        # automatic retries before giving up
  timeout: 10                           # per-request timeout in seconds

general_fallbacks:
  - primary-cluster: ["backup-cluster"] # if primary fails → try backup

context_window_fallbacks:
  - primary-cluster: ["backup-cluster"] # if message too long for primary → try backup
```

---

## 14. Troubleshooting FAQ

### Port 8000 or 8501 already in use

```bash
# Find what's using port 8000
lsof -i :8000

# Kill it (replace 12345 with the PID shown)
kill -9 12345
```

Or start on a different port:
```bash
PORT=8001 .venv/bin/python main.py
.venv/bin/streamlit run app.py --server.port 8502
```

---

### ModuleNotFoundError — missing package

You are using your system Python instead of the virtual environment.

Wrong:  `python main.py` / `streamlit run app.py`

Correct: `.venv/bin/python main.py` / `.venv/bin/streamlit run app.py`

Always prefix with `.venv/bin/` to use the correct interpreter.

---

### "You are sending unauthenticated requests to the HF Hub"

This appears the first time the base DeBERTa model is downloaded from Hugging Face. It is **safe to ignore**. The download continues automatically.

To suppress it, get a free token at https://huggingface.co/settings/tokens and set:
```bash
export HF_TOKEN="your_token_here"
```

---

### "MISMATCH: Reinit due to size mismatch — torch.Size([111]) vs torch.Size([113])"

This appears during evaluation and is **expected and harmless**. The base model's classifier head has 111 output labels; the fine-tuned version has 113 (we added 2 custom PII types). The weights load correctly.

---

### Dashboard shows dashes (—) for all stats

The FastAPI backend on port 8000 is not running. Start it first:
```bash
.venv/bin/python main.py
```
Wait for the `Uvicorn running on http://0.0.0.0:8000` line, then refresh the dashboard.

---

### Guardrail Evaluation tab shows old data

The dashboard reads from `data/evaluation_data.json`. Regenerate it:
```bash
.venv/bin/python scripts/evaluate.py
```
Then refresh the page. No server restart needed.

---

### Training fails with RuntimeError: CUDA out of memory

Training uses CPU by default. If PyTorch accidentally picks up a GPU, disable it:
```bash
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python scripts/train_pii_deberta.py
```

---

## 15. Glossary

| Term | Plain-English meaning |
|---|---|
| **FastAPI** | Python web framework — creates the API server running on port 8000 |
| **Uvicorn** | The web server process that runs FastAPI |
| **Streamlit** | Python framework for the browser admin console on port 8501 |
| **LiteLLM** | Library that connects to dozens of AI providers through one unified interface |
| **DeBERTa-v3** | The AI model that detects PII in text — runs 100% locally, no data leaves your machine |
| **PII** | Personally Identifiable Information — names, emails, phone numbers, IDs, medical data, etc. |
| **Fine-tuning** | Re-training an existing AI model on custom labelled data to improve task accuracy |
| **TPM** | Tokens Per Minute — how many tokens a provider allows per minute before rate-limiting |
| **RPM** | Requests Per Minute — how many API calls per minute are allowed |
| **TPR** | Tokens Per Request — maximum message length (context window) a model supports |
| **Cluster** | A named group of LLM providers that share load and act as each other's failover |
| **Virtual env** | An isolated Python installation — keeps this project's packages separate from your OS |
| **`.env` file** | A plain-text secrets file, loaded at startup, never committed to Git |
| **Complexity Tier** | LOW / MEDIUM / HIGH — how hard the system rates a request, used to pick the right model |
| **Macro F1** | Evaluation metric averaging F1 score across all PII entity types equally |
| **Confusion Matrix** | Table showing which PII types were correctly detected vs. confused with each other |

---

*Last updated July 2026 — FastAPI · LiteLLM · DeBERTa-v3 · Streamlit · PyTorch*
