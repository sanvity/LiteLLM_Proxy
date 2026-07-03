# Folder Directory Documentation

This document explains the organization and directories of the project, including why they exist, their responsibilities, interactions, and internal execution flows.

---

## 1. `config/`

### Purpose
The `config/` directory acts as the central storage folder for both static runtime configuration files and dynamic state parameter overrides.

### Why it Exists
It isolates configuration specifications from code files. By grouping configurations here, deployment pipelines can mount these settings as Kubernetes ConfigMaps or read/write settings across persistent volumes.

### Interactions with the Rest of the Project
- **`proxy/config.py`**: Locates and parses files here to read model properties.
- **`proxy/router.py`**: Reads `pii_guardrail_config.json` on startup to initialize memory flags, policies, and custom trained labels.
- **`proxy/app.py`**: Updates `pii_guardrail_config.json` via HTTP POST requests from the Streamlit UI or public APIs, persisting changes to disk.

### Internal Execution Flow
There is no code execution flow inside this directory as it only contains configurations:
1. `litellm_config.yaml`: Used for LiteLLM native parameters, custom callbacks, and custom guardrail setup.
2. `pii_guardrail_config.json`: Updated dynamically at runtime to toggles PII protection parameters.

---

## 2. `frontend/`

### Purpose
This directory contains static, client-side web application assets for the user interface dashboard.

### Why it Exists
It decouples user interfaces from backend services, allowing serverless hosting of the front-end dashboard on CDNs (like Vercel) while keeping the API gateway backend hosted on scalable application servers (like Render).

### Interactions with the Rest of the Project
- **`proxy/app.py`**: Serves as the target API gateway. The JavaScript code inside `frontend/index.html` fires HTTP requests to `/metrics`, `/v1/models`, `/v1/chat/completions`, and PII configuration endpoints.
- **Vercel Edge Network**: Reads `vercel.json` to route incoming client dashboard routes and reverse-proxy API requests back to Render to bypass CORS checks.

### Execution Flow
1. The user opens the frontend URL (e.g. Vercel deployment link).
2. The browser downloads and executes `index.html`.
3. The embedded JavaScript starts polling `/health` and `/metrics` from the API backend.
4. When a user submits queries, JavaScript sends JSON payloads to `/v1/chat/completions` on the gateway and displays latency, sanitized text, and responses.

---

## 3. `guardrails/`

### Purpose
This directory encapsulates the local token classification pipelines, remediation processors, and model fine-tuning handlers.

### Why it Exists
It acts as the core compliance module. By separating guardrails into its own folder, developers can maintain, run baseline evaluations, or train model scripts without altering the proxy routing microservice.

### Interactions with the Rest of the Project
- **`proxy/router.py`**: Invokes `DeBERTaPIIGuardrail.process` inside the completions execution loop.
- **`proxy/app.py`**: Spawns `deberta_pii_guardrail.py` as an out-of-process training subprocess when fine-tuning is triggered. It also calls `reload_model` on active guardrail pipelines once training finishes.

### Execution Flow
1. On start, the router imports `DeBERTaPIIGuardrail` from `guardrails/deberta_pii_guardrail.py`.
2. Upon prompt arrival, text is fed into the DeBERTa model pipeline to get entity token tags.
3. If matches exceed the threshold, remediation actions (MASK, BLOCK, or REWRITE) are applied.
4. If a training request arrives, the app spawns a subprocess: `python deberta_pii_guardrail.py --train <config_path>` which runs the Hugging Face Trainer fine-tuning loop and saves new weights to the `models/` directory.

---

## 4. `models/`

### Purpose
The `models/` directory houses the local fine-tuned model config files, pytorch binaries, and tokenizer vocabularies.

### Why it Exists
It keeps machine learning model weights local to the application rather than depending on external registries (like Hugging Face Hub) at runtime. This provides air-gapped security, zero external downloads during startup, and immediate access to custom parameters.

### Interactions with the Rest of the Project
- **`guardrails/deberta_pii_guardrail.py`**: Scans `models/finetuned-deberta` first. If `config.json` exists, it instantiates the token classification pipeline from these local weights; otherwise, it falls back to the base model.
- **`proxy/app.py`**: Reads `models/finetuned-deberta/config.json` on startup to detect if a custom model is active and set initial status flags.

### Internal Execution Flow
Like `config/`, this folder contains model artifacts and does not execute scripts. However, when the background fine-tuning subprocess finishes training, it atomically deletes the old folder contents and writes new weights here.

---

## 5. `proxy/`

### Purpose
The `proxy/` directory contains the complete FastAPI and LiteLLM microservice, including endpoints, config parsing logic, load-balancing router algorithms, and custom templates.

### Why it Exists
It represents the central application package. Grouping these modules into an OOP structure ensures clean imports, dependency injection, and isolates API serving logic from scripts and training processes.

### Interactions with the Rest of the Project
- **`main.py`**: The top-level startup script imports `LiteLLMProxyApp` from this package and starts the Uvicorn server.
- **`guardrails/deberta_pii_guardrail.py`**: Integrates with the proxy router via pre-call and post-call hooks to sanitize prompts and responses.
- **Streamlit Console (`app.py`)**: Interacts with the proxy endpoints to sync credit limits, preferences, metrics, and trigger training routines.

### Execution Flow
1. `main.py` imports `proxy` package and instantiates `LiteLLMProxyApp`.
2. `LiteLLMProxyApp` instantiates `ProxyConfig`, which parses `config.yaml`.
3. `LiteLLMProxyApp` instantiates `LiteLLMProxyRouter`, which sets up LiteLLM's `Router` and Redis connections.
4. FastAPI routes incoming completions requests to `LiteLLMProxyRouter.execute_chat_completion`.
5. The router classifies complexity, audits credit limits/utilizations, calls downstream LLM endpoints, updates usage counters, and returns responses.

---

## 6. `streamlit_app/`

### Purpose
This folder is structured specifically for running and deploying the Streamlit management console dashboard onto cloud hosting providers like Railway or Heroku.

### Why it Exists
It isolates Streamlit dependencies and setup configurations (like `Procfile` and `railway.toml`) from the main FastAPI server setup, allowing separate cloud deployments.

### Interactions with the Rest of the Project
- **`proxy` package**: Calls metrics, preference, and training routes on the deployed backend proxy service.
- **`synthetic_data.py`**: Imports the synthetic generation engine to generate balance training datasets in the model fine-tuning page.

### Execution Flow
1. The cloud host starts the service using `Procfile` command: `streamlit run app.py`.
2. The Streamlit runner starts the web dashboard on port 8501.
3. The dashboard executes layout cells, loads configuration fields, and runs interactive event loops connecting to the FastAPI proxy backend.
