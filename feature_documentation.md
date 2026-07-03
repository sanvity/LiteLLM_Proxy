# Feature Documentation

This document explains the core enterprise features implemented in the LiteLLM Proxy system, detailing their business requirements, implementations, data flows, and constraints.

---

## 1. Complexity-Aware Multi-Tier Routing

### Purpose
Intelligently classifies incoming prompts into complexity classes (`low`, `medium`, `high`) and routes them to appropriate cost-tier models.

### Business Requirement
Reduces operating costs by routing simple queries to cheap, fast models (e.g. 8B parameter models) while reserving expensive reasoning models (e.g. 70B parameter models) for complex math, coding, or logic tasks.

### Implementation
- Implemented in `LiteLLMProxyRouter.classify_prompt_complexity` and `execute_chat_completion`.
- Prompts are classified as `"high"` complexity if they require more than 8K context or contain coding, mathematical, or architectural keywords (e.g. `python`, `debug`, `solve`, `optimize`).
- Prompts are classified as `"medium"` complexity if they request summarization, formatting, or translation, or exceed a size threshold of 500 characters.
- The router calculates a utility score for each endpoint:
  `Utility Score = (tier_mismatch_penalty, cost_per_million, resource_utilization)`
- Applies a mismatch penalty (+5.0) if a high-complexity prompt is routed to a low-cost node, or if a low-complexity prompt is routed to an expensive node.

### Files Used
- [`config.yaml`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/config.yaml) (declares complexity tiers and costs)
- [`proxy/config.py`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/proxy/config.py) (parses configurations)
- [`proxy/router.py`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/proxy/router.py) (runs classification and routing)

### Execution Flow
1. A completion request arrives at `/v1/chat/completions`.
2. The router estimates request tokens and required context.
3. The prompt is classified into a complexity tier (`low`, `medium`, or `high`).
4. Available endpoints are evaluated, and the node with the lowest utility score is selected.
5. The request is dispatched to the chosen LLM backend.

### Input
- Messages list and requested completion token limits.

### Output
- Mapped completion response and the classified complexity string.

### Dependencies
- None (runs locally using keyword heuristics).

### Limitations
- Keyword heuristics may misclassify prompts that contain technical terms in casual contexts.

---

## 2. Tokens Per Request (TPR) Context Escalation

### Purpose
Calculates prompt size dynamically to prevent context overflow errors.

### Business Requirement
Avoids request failures by automatically escalating large prompts to high-capacity nodes.

### Implementation
- Implemented in `LiteLLMProxyRouter.estimate_request_tokens` and `execute_chat_completion`.
- Prompts are tokenized using `tiktoken` (falling back to character approximation: `len(text) // 4`).
- Required context is calculated as: `estimated_prompt_tokens + max_tokens`.
- Any endpoint with a TPR limit smaller than the required context is filtered out.
- If the prompt is too large for all standard nodes, the router relaxes the TPR constraint and escalates the request to the highest capacity backup node (e.g. Together AI Llama 3.3 70B with 128K context).

### Files Used
- [`proxy/router.py`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/proxy/router.py) (performs token checks and filters nodes)

### Execution Flow
1. A request arrives at `/v1/chat/completions`.
2. Prompts are tokenized to estimate the required context size.
3. Endpoints with insufficient TPR capacity are filtered out.
4. If all endpoints are filtered out, the request is escalated to the highest capacity backup node.

### Input
- Chat messages history and `max_tokens` settings.

### Output
- Sanitized response from the escalated node.

### Dependencies
- `tiktoken` (for token estimation).

### Limitations
- Character approximations may under- or over-estimate tokens for non-English languages.

---

## 3. Real-Time PII Guardrail Shield

### Purpose
Detects and redacts Personally Identifiable Information (PII) in prompts and responses.

### Business Requirement
Protects privacy and ensures regulatory compliance (e.g. GDPR, HIPAA) by preventing sensitive data (names, SSNs, credit cards) from being sent to external APIs.

### Implementation
- Implemented in `DeBERTaPIIGuardrail` inside `deberta_pii_guardrail.py`.
- Integrates with LiteLLM's `CustomGuardrail` middleware hooks: `async_pre_call_hook` (prompt sanitization) and `async_post_call_success_hook` (response sanitization).
- Uses a local DeBERTa-v3 model to classify PII entity spans.
- Applies remediation actions based on the configured policy:
  - **MASK**: Replaces sensitive spans with labels (e.g. `[SOCIAL_SECURITY_NUMBER]`).
  - **BLOCK**: Blocks the request and raises a 400 Bad Request error.
  - **REWRITE**: Sends the text to a local Ollama instance running Mistral to rewrite the text without PII.
  - **IGNORE**: Allows the text to pass through unchanged.

### Files Used
- [`guardrails/deberta_pii_guardrail.py`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/guardrails/deberta_pii_guardrail.py) (implements detection and remediation)
- [`config/pii_guardrail_config.json`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/config/pii_guardrail_config.json) (persists active policies)

### Execution Flow
1. A prompt is submitted to the gateway.
2. The pre-call hook runs token classification on the text.
3. If PII is found, the configured policy action is applied.
4. The sanitized prompt is sent to the LLM.
5. The post-call hook runs classification on the response and applies the policy.
6. The sanitized response is returned to the client.

### Input
- Raw text prompts or response strings.

### Output
- Sanitized text strings.

### Dependencies
- `transformers` (pipeline classification).
- Local PyTorch model files.
- Local Ollama service (optional, for REWRITE actions).

### Limitations
- Token classification models may occasionally produce false positives or false negatives on short or ambiguous strings.

---

## 4. Priority-Based Preference Routing & Credit Budgets

### Purpose
Allows users to configure model preferences and track credit consumption.

### Business Requirement
Enables priority-based model selection while protecting against budget overruns by automatically falling back to cheaper models.

### Implementation
- Implemented in `LiteLLMProxyRouter.execute_chat_completion` and `_update_success_metrics`.
- Admins configure a prioritized list of models and allocate a credit budget (in USD) for each.
- When enabled, the router selects the first preferred model that is within its budget, satisfies TPR limits, and is under resource limits.
- On request completion, the actual tokens consumed are used to update the model's accumulated spend:
  `cost = (total_tokens * cost_per_million) / 1,000,000`
- Once a model exceeds its budget, the router automatically cascades to the next model in the preference list.

### Files Used
- [`proxy/router.py`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/proxy/router.py) (evaluates budgets and updates spends)
- [`proxy/app.py`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/proxy/app.py) (exposes endpoints to sync configs)

### Execution Flow
1. A request is received.
2. The router evaluates the preference list.
3. Validates that the target model's accumulated spend is below its budget limit.
4. Routes the request to the preferred model.
5. Updates the model's accumulated spend on response completion.

### Input
- Request payload.

### Output
- Completed response.

### Dependencies
- None.

### Limitations
- Spend metrics are kept in memory and reset on server restart unless sync endpoints are integrated with persistent storage.

---

## 5. Asynchronous Supervised Fine-Tuning API

### Purpose
Enables supervised fine-tuning of the local DeBERTa model on custom datasets.

### Business Requirement
Allows organizations to train the PII classifier on domain-specific formats (e.g. internal customer codes) without manual script execution.

### Implementation
- Implemented in `LiteLLMProxyApp.run_training_in_background` and `train_deberta_model`.
- Admins submit training datasets and parameters via API.
- The backend unloads active inference models from memory to prevent out-of-memory crashes.
- Spawns the training loop as a subprocess:
  `python deberta_pii_guardrail.py --train <config_path>`
- Applies class weighting to downweight the background 'O' class and mitigate dataset imbalance.
- Automatically resumes from the latest checkpoint if interrupted.
- If Optuna is enabled, runs hyperparameter search trials first.
- Saves the fine-tuned weights and reloads the active inference pipelines.

### Files Used
- [`proxy/app.py`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/proxy/app.py) (manages background threads and routes)
- [`guardrails/deberta_pii_guardrail.py`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/guardrails/deberta_pii_guardrail.py) (implements training loops)

### Execution Flow
1. Admin submits a dataset to `/v1/deberta/train`.
2. The server spawns a background thread and unloads the inference pipelines.
3. A subprocess runs the Hugging Face Trainer fine-tuning loop.
4. Updates are written to `training_history.json`.
5. Upon successful completion, the subprocess exits.
6. The backend reloads the active inference pipelines with the new weights.

### Input
- Labeled training datasets and training parameters (epochs, batch size, learning rate).

### Output
- Status flags and loss logs.

### Dependencies
- PyTorch and Hugging Face transformers libraries.

### Limitations
- Running fine-tuning on resource-constrained hosting services (like Render free tier CPU) can be slow and may hit memory limits.

---

## 6. Diversity-Driven Synthetic Dataset Generation

### Purpose
Generates balanced, diversity-driven datasets to train PII models.

### Business Requirement
Reduces the time and cost of collecting and labeling training data.

### Implementation
- Implemented in `SyntheticDataEngine` inside `synthetic_data.py`.
- Generates mock values using Faker or custom regex patterns.
- Distractors (hard negatives) are generated using character mutation algorithms.
- Sentences are generated in batches using parallel threads.
- Duplicates are filtered out using sliding similarity checks.
- Spans are validated in a post-generation pass to ensure start/end offsets match the text values.

### Files Used
- [`synthetic_data.py`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/synthetic_data.py) (implements generation engine)
- [`run_synthetic_engine.py`](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/run_synthetic_engine.py) (CLI runner)

### Execution Flow
1. Target labels and domains are configured.
2. The engine groups generation tasks and starts parallel threads.
3. Prompts instruct the LLM to write natural sentences with `[PII]` placeholders.
4. Mock values are generated and inserted into the placeholders.
5. Submitting checks filter out near-duplicates.
6. A post-generation validation pass drops invalid samples.
7. Validated samples are saved as a JSON dataset.

### Input
- Target labels, count limits, and format domains.

### Output
- Validated JSON dataset.

### Dependencies
- `exrex` and `faker` libraries.

### Limitations
- Relies on external API providers to write the carrier sentences, which may hit rate limits.
