# System Architecture Overview

This document provides a comprehensive overview of the **LiteLLM Load-Balancing & Routing Proxy** architecture. It details the separation of concerns, communication flows, data paths, and model training pipelines.

---

## High-Level System Components

The project consists of three main logical blocks:
1. **Intelligent Gateway Proxy (FastAPI + LiteLLM)**: Handles incoming requests, estimates tokens, processes guardrails, and routes prompts to optimal backend models.
2. **PII Guardrail Shield (Transformers/DeBERTa + Hugging Face Trainer + Optuna)**: Running locally, this block detects PII, runs masking/blocking/rewriting remediation, and hosts the fine-tuning endpoint.
3. **Interactive Control Console (Streamlit)**: Serves as a developer and administrator dashboard to configure prioritization, set budgets, generate synthetic data, and trigger model fine-tuning.

## System Overview
This is a PII-aware LLM proxy system with a control dashboard, a routing layer, and a guardrail model that's continuously fine-tuned.
1. Streamlit Console (Dashboard Layer)
A Streamlit-based UI (app.py) serves as the admin/control panel. It includes a drag-and-drop interface (via streamlit-sortables) for reordering preferences. From this dashboard, users can:

-Fetch the current preference configuration (GET /preference-config)
-Update PII detection settings (POST /ui/pii-config)
-Trigger training of the DeBERTa PII model (POST /ui/train-deberta)

All three actions go through the FastAPI Proxy.

2. FastAPI Proxy Service (Core Routing Layer)
This is the brain of the system, made up of three components:

-LiteLLMProxyApp — the main app that receives requests from the dashboard
-LiteLLMProxyRouter — decides where each request/query should be routed
-ProxyConfig — manages configuration settings

The App delegates routing decisions to the Router and configuration to Config.

3. PII Guardrail Service
Before any query reaches an external LLM, it passes through a DeBERTa-based PII guardrail (deberta_pii_guardrail.py) that screens for sensitive personal information. There's also a Supervised Fine-Tuning (SFT) loop running asynchronously — meaning the App can kick off model retraining jobs in the background without blocking the main request flow, likely using the training data/config gathered from the dashboard.

4. External Services (Where completions actually happen)
Once a query passes the guardrail, the Router sends it to one of several LLM providers depending on routing logic:

Groq, Cerebras, and Together AI — cloud-hosted completion APIs (likely load-balanced or chosen based on cost/latency/availability)
Local Ollama (Mistral) — used as a fallback option, or to rewrite/sanitize a query (e.g., if PII was detected and the query needs reformulating before going to a cloud provider)
Redis — a cache used to track rate limits across these API calls, preventing the system from exceeding provider quotas

## Request Execution Flow (Completions)

When a client submits a prompt to the `/v1/chat/completions` endpoint, the execution passes through several validation hooks and routing decisions:

1. Client Request
A client sends a chat completion request — model name, messages, and parameters — to the FastAPI controller. The controller immediately validates the payload against a ChatCompletionRequest Pydantic model to make sure the shape of the request is correct before doing anything else.

2. Pre-Call PII Check (Input Guardrail)
Before the request goes anywhere near an LLM, it's passed to the DeBERTa PII Guardrail, which scans the messages for sensitive personal information (names, emails, phone numbers, etc.) using entity-span detection.
If PII is found, one of three things happens depending on how the guardrail is configured:

-BLOCK — The request is rejected outright. The guardrail raises an error, and the client gets a 400 Bad Request. Nothing reaches the LLM.
-MASK — The detected PII is simply replaced with placeholder labels (e.g., [NAME], [EMAIL]) and the request continues.

If no PII is found (or the guardrail is disabled), the messages pass through untouched.

3. Routing Decision
Once the messages are clean, the FastAPI controller hands off to the Proxy Router, which decides where the completion should actually be executed. It does this in a few steps:
First, it estimates the size of the request — counting tokens via tiktoken and factoring in max_tokens to know how much context capacity is needed. Then it classifies the request's complexity as LOW, MEDIUM, or HIGH based on content and context length.
From there, it picks a node using one of two strategies. If priority routing is enabled and the account is within its credit limit, it selects the highest-priority preferred provider from a preference list. Otherwise, it falls back to standard load balancing — finding all providers that support the requested model, checking current rate-limit usage (TPM/RPM, tracked via Redis or an in-memory sliding window), and scoring candidates by lowest cost, mismatch, and utilization to pick the best fit.

4. Dispatch & Completion
The chosen request is sent to the target LLM provider through LiteLLM's router. The response comes back, and the router logs usage — updating token/request-per-minute counters and deducting credit spend — before handing the result back to the FastAPI controller.

5. Post-Call PII Check (Output Guardrail)
Before sending anything back to the client, the output is also run through the PII guardrail. If PII shows up in the generated response — the LLM could leak something it wasn't supposed to — the same BLOCK/MASK/REWRITE logic applies, this time on the response text.

6. Final Response
Once the response is confirmed clean, the client gets back a 200 OK with the sanitized completion.

## Model Fine-Tuning Execution Flow

The platform supports local fine-tuning of the DeBERTa model to recognize new, domain-specific custom PII entities (e.g. `patient_id`, `employee_code`). Due to memory limits, this occurs out-of-process in a background subprocess.

1. Admin Kicks Off Training
An admin submits custom training samples (as JSON) along with hyperparameters through the Streamlit Console. The UI forwards this as a POST /ui/train-deberta request to the FastAPI controller.
The controller immediately sets the training status to "training" and spawns a background thread to handle the heavy lifting — then responds right away with a 200 OK confirming training has been kicked off. This keeps the UI responsive instead of making the admin wait for the whole training run to finish.

2. Preparing for Training
Inside that background thread, the controller first unloads the live inference pipelines from memory and runs garbage collection — freeing up resources (likely GPU/RAM) so the training process has room to work without conflicting with whatever model is currently serving live requests.
It then spawns a separate subprocess — literally launching python deberta_pii_guardrail.py --train with a temporary config file — so training runs in its own isolated process rather than inside the main API server.

3. Inside the Training Subprocess
Once running, the subprocess:
Loads the base or existing local model weights as the starting point. It then calculates class weights to correct for the natural imbalance in the data — since most tokens in any text are not PII (labeled "O"), the model needs to be nudged not to just predict "not PII" every time. It also aligns tokenization offsets, which is necessary to make sure the PII labels line up correctly with the sub-word tokens the model actually sees.
If Optuna hyperparameter search is enabled, the subprocess runs multiple tuning trials to find the hyperparameter combination that minimizes validation loss, then overrides the final training config with whatever won.
After that, it initializes a WeightedTrainer (using the class weights calculated earlier) and loops through training epochs — running the actual Hugging Face training steps and logging progress to training_history.json after each one, so the admin can track progress in near real-time.

4. Wrapping Up Training
Once training finishes, the subprocess unloads the training pipeline, saves the new tokenizer and model weights to a temporary folder, then does an atomic replace of the live model directory (models/finetuned-deberta) — swapping in the new model in one clean step rather than overwriting files incrementally, which avoids ending up with a half-updated model if something goes wrong. The subprocess then exits cleanly.

5. Reloading the Live System
Back in the FastAPI controller, it reloads the live model pipelines across all active worker instances so the new model actually starts serving requests. It saves any new custom labels to pii_guardrail_config.json, and updates the status to "completed".

6. Checking Progress
At any point, the admin can poll GET /ui/train-deberta/status from the UI, and the controller returns the current status along with the training history logs — so progress can be watched live rather than just waiting blindly for a done signal.
