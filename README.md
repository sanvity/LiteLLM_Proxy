# 🛡️ LiteLLM Gateway Console — Production-Ready AI Proxy

An enterprise-grade AI gateway with **intelligent LLM routing**, **real-time PII shielding**, and a **built-in Streamlit dashboard** — accessible from a single URL in any browser, no local setup required.

Built with **FastAPI**, **LiteLLM**, **DeBERTa-v3**, and **Streamlit**, deployed as a single Docker container.

---

## ☁️ One-Click Deploy

Deploy to your preferred cloud platform by clicking a button below. Set your API key secrets in the platform dashboard after deployment.

### Railway (Recommended)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/new?template=https://github.com/sanvity/LiteLLM_Proxy)

> After deploy: Go to **Variables** → add `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `TOGETHERAI_API_KEY`

### Render
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/sanvity/LiteLLM_Proxy)

> After deploy: Go to **Environment** → add your API keys as secret variables

### Fly.io
```bash
# Install Fly CLI, then:
fly launch --no-deploy
fly secrets set GROQ_API_KEY=gsk_... CEREBRAS_API_KEY=csk_... TOGETHERAI_API_KEY=...
fly deploy
```

---

## 🔑 Required Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Groq API key ([console.groq.com](https://console.groq.com)) | ✅ |
| `CEREBRAS_API_KEY` | Cerebras API key ([cloud.cerebras.ai](https://cloud.cerebras.ai)) | ✅ |
| `TOGETHERAI_API_KEY` | Together AI key ([api.together.ai](https://api.together.ai)) | ✅ |
| `MISTRAL_API_KEY` | Mistral API key (optional) | ⬜ |
| `PORT` | HTTP port (auto-set by cloud platform) | auto |
| `OLLAMA_API_BASE` | Local Ollama URL (local only) | ⬜ |

See [`.env.example`](.env.example) for a full template.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🔀 **Intelligent Routing** | `usage-based-routing` — automatically balances load across Groq, Cerebras, Together AI |
| 🛡️ **PII Shield (DeBERTa-v3)** | Real-time detection of 10+ PII entity types with BLOCK / MASK / REWRITE controls per entity |
| 📊 **Live Dashboard** | Streamlit UI served at `/` — no separate port, works on any cloud |
| ⚡ **OpenAI-Compatible API** | Drop-in replacement for any OpenAI SDK client |
| 🔁 **Auto Fallbacks** | `primary-cluster → backup-cluster` with context-window escalation |
| 🏗️ **Single Container** | One Docker image, one port, one URL |

---

## 🏗️ Architecture

```
Browser → https://myapp.com/           (single public URL)
              │
              ▼
     ┌─────────────────────┐
     │   FastAPI Gateway   │  port $PORT (e.g. 8000)
     │                     │
     │  ┌───────────────┐  │  HTTP + WebSocket reverse proxy
     │  │  Streamlit UI │◄─┼─── / (root) and /_stcore/* paths
     │  │  (port 8501)  │  │
     │  └───────────────┘  │
     │                     │
     │  ┌───────────────┐  │
     │  │  LLM Router   │◄─┼─── /v1/chat/completions
     │  │  DeBERTa PII  │  │
     │  └───────────────┘  │
     └─────────────────────┘
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
 Groq     Cerebras   Together AI
```

---

## 📁 Directory Structure

```text
LiteLLM_Proxy/
├── Dockerfile           # Multi-stage build (model pre-cache, slim runtime)
├── .dockerignore        # Keeps secrets out of the image
├── railway.toml         # Railway deployment config
├── render.yaml          # Render.com deployment blueprint
├── fly.toml             # Fly.io app configuration
├── Procfile             # Heroku/Railway fallback entrypoint
├── .env.example         # Environment variable template (safe to commit)
├── config.yaml          # LiteLLM routing, capacity, and fallback settings
├── requirements.txt     # Python production dependencies
├── main.py              # Unified entrypoint: spawns Streamlit + starts FastAPI
├── app.py               # Streamlit dashboard frontend
├── test_proxy.py        # Automated unit & integration test suite
└── proxy/               # FastAPI package
    ├── app.py           # Routes, Streamlit reverse proxy, PII config endpoints
    ├── router.py        # Token estimation, routing, DeBERTa PII guardrail
    ├── config.py        # Pydantic config models
    └── guardrails/      # DeBERTa-v3 PII detection engine
```

---

## 🖥️ Local Development

### 1. Clone & Install

```bash
git clone https://github.com/sanvity/LiteLLM_Proxy.git
cd LiteLLM_Proxy

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Start the Application

```bash
python main.py
```

The app will:
1. Spawn the Streamlit dashboard on internal port 8501
2. Wait for Streamlit to be ready
3. Start FastAPI on port 8000

Then visit **http://localhost:8000** — the full UI loads in your browser.

---

## 🐳 Local Docker Build

```bash
docker build -t litellm-gateway .
docker run -p 8000:8000 --env-file .env litellm-gateway

# Visit http://localhost:8000
```

---

## 🧪 Running Tests

```bash
python -m unittest test_proxy.py
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Streamlit dashboard (proxied) |
| `/health` | GET | Liveness probe — returns `{"status": "healthy"}` |
| `/metrics` | GET | Real-time routing and token metrics |
| `/v1/models` | GET | List virtual and physical models |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completion |
| `/ui/pii-config` | GET/POST | Read/update PII guardrail configuration |
| `/preference-config` | GET/POST | Read/update model preference routing |
| `/old-ui` | GET | Legacy HTML dashboard (debugging) |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "primary-cluster",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```
