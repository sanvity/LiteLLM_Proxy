# LiteLLM Load-Balancing & Routing Proxy Server

An enterprise-grade, class-based (OOP) microservice designed for intelligent load balancing, dynamic routing, rate-limit management (TPM/RPM), context-window filtering (TPR), and automatic fallbacks across multiple open-source LLM providers. 

Built using **FastAPI**, **LiteLLM**, and **Pydantic**, this server acts as a unified gateway for open-source AI, abstracting complex backend topologies (Groq, Cerebras, Together AI, Ollama, and Mistral) into a single, high-availability, OpenAI-compatible API.

---

## Key Features

1. **Class-Based (OOP) Architecture**: Fully encapsulated microservice components (Config, Router Engine, and FastAPI Controller) designed for clean dependency injection and seamless microservice integration.
2. **Unified Virtual Models**: Abstracts backend providers under two clean, logical groups:
   - `oss-chat-fast`: Extremely fast, lightweight open-source models (e.g. Llama 3 8B, Mistral Nemo) served across Cerebras, Groq, Together AI, or local Ollama.
   - `oss-chat-premium`: Large-context, advanced reasoning models (e.g. Llama 3.1 70B, Mistral Large) served across Groq, Mistral, or Together AI.
3. **Dynamic Load Balancing**: Uses `latency-based-routing` to automatically monitor provider response times and route requests to the fastest active deployment.
4. **Token Per Request (TPR) Filtering**: Automatically estimates prompt token length (`tiktoken`) to screen out endpoints that cannot support the request size, gracefully escalating to larger context windows when needed.
5. **High Availability & Fallbacks**: Implements automatic multi-provider fallback chains. If Groq encounters rate limits (`429`) or Cerebras goes down, the proxy instantly reroutes to Together AI or a local Ollama deployment.
6. **Built-in Mock Sandbox Mode**: Includes a comprehensive local testing sandbox that simulates realistic responses without calling downstream APIs, ideal for CI/CD and cost-free local verification.
7. **Kubernetes-Ready**: Includes pre-configured `/health` (liveness/readiness probes) and `/metrics` (for real-time observability).

---

## System Architecture

```
                      +-----------------------------+
                      |       Client Request        |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   FastAPI Proxy Controller  |
                      |        (proxy/app.py)       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Routing & Token Manager   |
                      |       (proxy/router.py)     |
                      +-------+--------------+------+
                              |              |
           (Token Estimation) |              | (Custom TPR Filtering & Selection)
                              v              v
                      +---------------+     +--------------------------------+
                      | Tiktoken cl100|     |  OOP LiteLLM Router Wrapper    |
                      | k / Fallback  |     |   - Simple-Shuffle/Latency     |
                      +---------------+     |   - Fallbacks & Retries        |
                                            +---------------+----------------+
                                                            |
                 +-----------------+----------+-------------+------------+-----------------+
                 |                 |          |                          |                 |
                 v                 v          v                          v                 v
          +------------+     +-----------+  +-------------+       +------------+     +-----------+
          |    Groq    |     |  Cerebras |  | Together AI |       |   Ollama   |     |  Mistral  |
          | Llama3 70B |     | Llama3 8B |  | Mixtral 8x7B|       | Llama3 (L) |     |  Large    |
          +------------+     +-----------+  +-------------+       +------------+     +-----------+
```

---

## Directory Structure

```text
LiteLLM_Proxy/
├── config.yaml          # LiteLLM YAML routing, capacity and fallback settings
├── requirements.txt     # Python production dependencies
├── main.py              # Microservice server startup script
├── test_proxy.py        # Complete automated OOP unit & integration test suite
├── .env                 # Local active credentials (includes sandbox defaults)
├── .env.example         # Template environment file
└── proxy/               # Class-based python package core
    ├── __init__.py      # Package definitions & exports
    ├── config.py        # OOP Pydantic configurations
    ├── router.py        # OOP Token estimation and route optimizer wrapper
    └── app.py           # Class-based FastAPI controllers & OpenAI schemas
```

---

## Installation & Setup

### 1. Clone & Navigate to Workspace
```bash
cd /Users/sanvijain/EY_DataAndAI/LiteLLM_Proxy
```

### 2. Set Up a Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Credentials
Edit the active `.env` file to insert your specific provider API keys:
```env
GROQ_API_KEY=gsk_...
CEREBRAS_API_KEY=csk_...
TOGETHERAI_API_KEY=...
MISTRAL_API_KEY=...
OLLAMA_API_BASE=http://localhost:11434
```
*Note: If environment variables remain set to mock placeholders (like `mock_groq_key`), the proxy will automatically operate in the zero-cost Mock Sandbox mode.*

---

## Running the Application

### Start the Proxy Server
Launch the ASGI server locally using:
```bash
python main.py
```
This boots up the proxy on `http://0.0.0.0:8000`.

### Health & Metrics Endpoints
- **Liveness Probe**: `GET http://localhost:8000/health`
- **Observability metrics**: `GET http://localhost:8000/metrics`
- **OpenAI Model Catalog**: `GET http://localhost:8000/v1/models`

---

## Verifying the Proxy

We provide a complete automated unit and integration test suite that tests configurations, token calculation, TPR routing, mock sandbox modes, and full HTTP integrations:

```bash
python -m unittest test_proxy.py
```

### Client Request Example (cURL)
You can call the proxy using any standard OpenAI-compatible client library (e.g. LangChain, OpenAI Python SDK) or simple cURL requests:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "oss-chat-fast",
    "messages": [
      {"role": "user", "content": "Explain the concept of quantum computing."}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```
