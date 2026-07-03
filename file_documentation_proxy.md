# File Documentation: Proxy Package

This document provides detailed class, function, parameter, and endpoint-level documentation for all components in the `proxy` directory and the main entry point.

---

# main.py

## Purpose
Acts as the production-ready startup and execution script for the Uvicorn ASGI server hosting the LiteLLM Proxy service.

## Responsibilities
- Bootstraps the application configuration and logs.
- Loads env variables from `.env`.
- Dynamically imports and instantiates the `LiteLLMProxyApp`.
- Binds and runs the FastAPI instance under Uvicorn.

## Dependencies
- `uvicorn` (ASGI Server)
- `dotenv` (Environment Loader)
- `logging` (Log Handler)
- `proxy` (Application Package)

## Imports
- `import os`: Used to fetch target host and port env vars.
- `import uvicorn`: Used to run the FastAPI app instance.
- `import logging`: Configures logging formats for the proxy service.
- `from dotenv import load_dotenv`: Loads environment settings.

## Functions

### `main`

#### Purpose
Executes environment loading, sets up logging, builds the FastAPI instance, and starts the ASGI server.

#### Parameters
None

#### Return Value
None

#### Exceptions
- Logs a `critical` event and exits with code `1` if imports, config files, or Uvicorn bindings fail.

#### Internal Logic / Algorithm
1. Configures logging with format `%(asctime)s [%(levelname)s] %(name)s: %(message)s` directed to `StreamHandler` (stdout).
2. Calls `load_dotenv()`.
3. Reads `PORT` (defaults to `8000`) and `HOST` (defaults to `"0.0.0.0"`).
4. Imports `LiteLLMProxyApp` locally to defer import errors until logging is active.
5. Instantiates `LiteLLMProxyApp` with `"config.yaml"`.
6. Retrieves the FastAPI instance via `proxy_service.get_app()`.
7. Starts `uvicorn.run(...)` synchronously.

#### Business Logic
N/A

#### Database Operations
None

#### External API Calls
None

#### AI Model Calls
None

#### Side Effects
Modifies system logging handler streams and binds a network socket on the specified host and port.

#### Functions Called
- `dotenv.load_dotenv()`
- `proxy.LiteLLMProxyApp`
- `proxy.LiteLLMProxyApp.get_app`
- `uvicorn.run`

#### Functions Calling This
- Entry point block: `if __name__ == "__main__": main()`

#### Example Usage
```bash
python main.py
```

---

# proxy/__init__.py

## Purpose
Exposes the public class API interface for the `proxy` package.

## Responsibilities
- Exports `ProxyConfig`, `ModelEndpointConfig`, `LiteLLMProxyRouter`, and `LiteLLMProxyApp` for clean external imports.

## Imports
- `from .config import ProxyConfig, ModelEndpointConfig`: Import configuration classes.
- `from .router import LiteLLMProxyRouter`: Import routing engine.
- `from .app import LiteLLMProxyApp`: Import FastAPI container.

---

# proxy/config.py

## Purpose
Defines the Pydantic data schemas for physical model endpoints and implements YAML configuration parsers.

## Responsibilities
- Declares the `ModelEndpointConfig` schema (which includes limits, costs, and complexity tiers).
- Parses `config.yaml` to configure the endpoints list and fallbacks.
- Auto-resolves env variables formatted as `os.environ/KEY`.
- Determines complexity tiers based on price boundaries.

## Dependencies
- `yaml` (YAML Loader)
- `pydantic` (Data Validation)

## Imports
- `import os`: Reading environment variables.
- `import yaml`: Parsing configuration yaml.
- `import logging`: Package logging actions.
- `from typing import List, Dict, Any, Optional`: Declaring type signatures.
- `from pydantic import BaseModel, Field`: Defining OOP config schemas.

## Classes

### `ModelEndpointConfig`

#### Purpose
Stores properties of a specific downstream LLM model backend, serving as an OOP model catalog entry.

#### Constructor
Inherits from Pydantic `BaseModel`.

#### Attributes
- `model_name` (`str`): Logical virtual model name (e.g. `primary-cluster`, `backup-cluster`).
- `model` (`str`): Downstream physical model path (e.g. `groq/llama-3.1-8b-instant`).
- `api_key` (`Optional[str]`): Resolved API key string.
- `api_base` (`Optional[str]`): Resolved endpoint URL (for local Ollama).
- `rpm` (`int`): Requests Per Minute rate limit (default: `1000`).
- `tpm` (`int`): Tokens Per Minute rate limit (default: `100000`).
- `tpr` (`int`): Tokens Per Request context window capacity (default: `8192`).
- `cost_per_million` (`float`): USD pricing per million tokens (default: `0.1`).
- `max_tokens` (`Optional[int]`): Hard limit on generated completion tokens.
- `complexity_tier` (`Optional[str]`): Complexity rating (`low`, `medium`, or `high`).

#### Relationships with Other Classes
- Instantiated by `ProxyConfig` during configuration parsing.
- Used by `LiteLLMProxyRouter` to score suitability and construct LiteLLM routing lists.

---

### `ProxyConfig`

#### Purpose
Loads, parses, and validates the configuration file, converting properties into `ModelEndpointConfig` instances.

#### Constructor
`__init__(self, config_path: str = "config.yaml")`
- Sets default properties: routing strategy to `"simple-shuffle"`, retry attempts to `3`, timeout to `10`.
- Invokes `load_config()`.

#### Attributes
- `config_path` (`str`): Filepath to YAML settings.
- `endpoints` (`List[ModelEndpointConfig]`): Parsed physical model list.
- `routing_strategy` (`str`): Router load-balancing setting.
- `fallback_policy` (`str`): Failover settings (default: `"retry_next_suitable"`).
- `num_retries` (`int`): Allowable retry sweeps.
- `timeout` (`int`): HTTP request timeouts.
- `context_window_fallbacks` (`List[Dict[str, List[str]]]`): Context overflow failover lists.
- `general_fallbacks` (`List[Dict[str, List[str]]]`): General failover lists.

#### Methods

##### `load_config`

###### Purpose
Reads the YAML settings file, extracts parameters, resolves env keys, and builds `ModelEndpointConfig` objects.

###### Parameters
None

###### Return Value
None

###### Exceptions
- Raises `FileNotFoundError` if the config path does not exist.
- Raises exceptions if YAML parsing fails.

###### Internal Logic / Algorithm
1. Reads `config_path`.
2. Extracts general router settings (`routing_strategy`, `num_retries`, `timeout`).
3. Extracts `context_window_fallbacks` and `general_fallbacks`.
4. Loops through `model_list`:
   - Checks if `api_key` and `api_base` match `os.environ/VAR_NAME` and replaces them with `os.environ.get("VAR_NAME")`.
   - If `complexity_tier` is not explicitly set, applies the fallback rules:
     - `cost >= 0.50` -> `"high"`
     - `cost >= 0.04` -> `"medium"`
     - `cost < 0.04` -> `"low"`
   - Instantiates `ModelEndpointConfig` and appends it to `self.endpoints`.

###### Business Logic
Applies automatic pricing thresholds to classify downstream models into reasoning classes (High vs Low) if not explicitly set.

###### Side Effects
Modifies local config state fields.

###### Functions Called
- `yaml.safe_load`
- `os.environ.get`
- `ModelEndpointConfig`

---

##### `get_endpoints_for_model`

###### Purpose
Filters the loaded endpoints to return those matching a virtual model name.

###### Parameters
- `virtual_model_name` (`str`): The name of the virtual model (e.g. `"primary-cluster"`).

###### Return Value
- `List[ModelEndpointConfig]`: Matched physical configs.

---

##### `to_litellm_model_list`

###### Purpose
Converts parsed endpoints to list dictionaries required by the LiteLLM Router initialization.

###### Parameters
None

###### Return Value
- `List[Dict[str, Any]]`: Mapped config entries with keys `model_name` and `litellm_params`.

---

# proxy/router.py

## Purpose
Core routing and token management engine of the gateway proxy, wrapping LiteLLM's `Router` and tracking rate limits.

## Responsibilities
- Instantiates LiteLLM's `Router` with configuration parameters.
- Integrates Redis for sliding-window token rate limits.
- Classifies prompt complexity based on context lengths and semantic reasoning indicators.
- Scores suitable endpoints using a utility function (mismatch penalty, pricing, and system load).
- Applies fallback routing cascades (intra-cluster and inter-cluster) on completion failures.
- Estimates prompt token requirements via `tiktoken`.
- Integrates the DeBERTa PII Guardrail pre-call and post-call hooks.

## Dependencies
- `litellm` (Core Router)
- `tiktoken` (Token Estimator)
- `redis` (Centralized limit cache)
- `httpx` (API Requests client)

## Imports
- `import time`, `import logging`, `import threading`, `import httpx`
- `from typing import List, Dict, Any, Optional, Tuple`
- `import tiktoken`, `import litellm`
- `from litellm import Router`
- `from .config import ProxyConfig, ModelEndpointConfig`

## Classes

### `LiteLLMProxyRouter`

#### Purpose
Executes prompt classification, load-balancing audits, downstream completions dispatching, failover tracking, and PII guardrail evaluations.

#### Constructor
`__init__(self, config: ProxyConfig)`
- Disables LiteLLM global telemetry and sets `drop_params = True`.
- Initializes LiteLLM's `Router` with parameters and fallbacks from config.
- Establishes a threading lock for metrics.
- Connects to Redis via `REDIS_HOST` (default: `"localhost"`) and `REDIS_PORT` (default: `6379`). Sets up memory cache simulation if Redis is unreachable.
- Loads tiktoken encoding `"cl100k_base"`.
- Loads PII configs and sets up default credit budgets.

#### Attributes
- `config` (`ProxyConfig`): Active config settings.
- `router` (`Router`): LiteLLM router instance.
- `metrics_lock` (`threading.Lock`): Mutex for local thread safety.
- `metrics` (`Dict[str, Any]`): Metrics counters (requests, tokens, fallback events).
- `redis_client` (`Optional[redis.Redis]`): Active Redis connection or None.
- `tokenizer` (`Optional[tiktoken.Encoding]`): Active tiktoken instance.
- `accumulated_spend` (`Dict[str, float]`): Tracked billing per model.
- `credit_limits` (`Dict[str, float]`): Cost spending caps.
- `routing_logs` (`List[Dict[str, Any]]`): Log buffer for dashboard visualization.

#### Methods

##### `load_pii_guardrail_config`

###### Purpose
Reads PII configuration JSON, updating enabled states, default actions, label mappings, and custom labels.

---

##### `save_pii_guardrail_config`

###### Purpose
Writes PII config fields to `config/pii_guardrail_config.json`.

---

##### `log_event`

###### Purpose
Logs messages and stores them in a 50-entry rolling buffer for dashboard diagnostics.

---

##### `_get_pii_guardrail`

###### Purpose
Helper to import and instantiate `DeBERTaPIIGuardrail` lazily.

---

##### `_sanitize_response`

###### Purpose
Applies post-call PII guardrails on generated LLM completions content.

###### Parameters
- `response` (`Dict[str, Any]`): Downstream completions response.
- `guardrailed_query` (`Optional[str]`): Pre-call sanitized query context.
- `bypass_guardrails` (`bool`): Toggle flag.

###### Return Value
- `Dict[str, Any]`: Sanitized completions payload.

###### Internal Logic
1. Returns response immediately if `pii_enabled = False` or `bypass_guardrails = True`.
2. Extracts content from `choices[0].message.content`.
3. Runs the active DeBERTa guardrail process.
4. If output contains blocked entities, replaces content with `[Response suppressed: model output contained blocked PII]`.
5. If masked/rewritten, sets content to the cleaned string.

---

##### `_prune_usage_history`

###### Purpose
Cleans memory-cache telemetry entries older than 60 seconds (sliding window).

---

##### `track_request_usage`

###### Purpose
Logs token consumption of a finished completion request.

###### Parameters
- `physical_model` (`str`): Target model name.
- `tokens` (`int`): Token size.

###### Internal Logic
- **Redis Case**: Uses sorted sets (`litellm:tpm:{model}` and `litellm:rpm:{model}`) via `zadd` with a timestamp-based key structure. Expire indices are set to 65 seconds.
- **Memory Case**: Appends timestamp, model name, and tokens to `self.usage_history`.

---

##### `get_endpoint_usage`

###### Purpose
Calculates TPM and RPM rates for a given physical model in the last 60 seconds.

###### Parameters
- `physical_model` (`str`): Model path.

###### Return Value
- `Tuple[int, int]`: Returns calculated `(tpm, rpm)`.

###### Internal Logic
- **Redis Case**: Truncates timestamps using `zremrangebyscore` and retrieves card lists. Parses tokens by splitting `timestamp:tokens` strings.
- **Memory Case**: Truncates history list and sums properties.

---

##### `estimate_tokens`

###### Purpose
Estimates string token length.

###### Parameters
- `text` (`str`): Target string.

###### Return Value
- `int`: Token count.

###### Internal Logic
- Returns `len(tokenizer.encode(text))` if tiktoken is active; otherwise, returns character approximation: `len(text) // 4`.

---

##### `estimate_request_tokens`

###### Purpose
Estimates overall token length of message history.

###### Parameters
- `messages` (`List[Dict[str, str]]`): Message payloads.

###### Return Value
- `int`: Total estimated request tokens.

---

##### `classify_prompt_complexity`

###### Purpose
Classifies prompt complexity into `'low'`, `'medium'`, or `'high'` based on context size and semantic checks.

###### Parameters
- `messages` (`List[Dict[str, str]]`): Request messages.
- `required_context` (`int`): Combined prompt/response context.

###### Return Value
- `str`: Complexity string.

###### Internal Logic
1. Any required context size > 8192 returns `"high"`.
2. Checks user message text for keywords:
   - High complexity: Coding languages, `"algorithm"`, `"refactor"`, `"optimize"`, `"math"`, `"logic"`, `"architecture"`, `"step by step"`. Also checks for code block ticks (```) or braces.
   - Medium complexity: `"summar"`, `"report"`, `"translate"`, `"email"`, `"explain"`.
3. If high-complexity indicator count >= 2 or (count >= 1 and context > 2048), returns `"high"`.
4. If medium-complexity count >= 1 or context > 1024 or length > 500, returns `"medium"`.
5. Else, returns `"low"`.

---

##### `execute_chat_completion`

###### Purpose
Performs prompt validation, complexity classification, model routing suitability scoring, LiteLLM completion execution, usage tracking, and fallback handling.

###### Parameters
- `model` (`str`): Virtual model name (e.g. `"primary-cluster"`).
- `messages` (`List[Dict[str, str]]`): Prompts list.
- `**kwargs` (`Any`): Downstream inference parameters.

###### Return Value
- `Dict[str, Any]`: Completed response.

###### Exceptions
- Raises `ValueError` if a PII violation blocks the request pre-call.
- Raises `ValueError` if no active endpoints are found.
- Raises `RuntimeError` if all backends in the failover chain fail.

###### Internal Logic
1. **Pre-Call Guardrail**: If PII is enabled, processes input messages. If blocked, throws `ValueError`. If MASK/REWRITE actions occur, updates the message content copy.
2. **Context Check**: Estimates request tokens.
3. **Complexity Check**: Runs `classify_prompt_complexity`.
4. **Endpoint Selection**:
   - **Preference Case**: If preference routing is active, scans the preferred models. Checks credit limit constraints (`accumulated_spend < credit_limits`), TPR capacity (`required_context <= ep.tpr`), and TPM/RPM limits. Selects the first matching node.
   - **Standard Case**: Filters the virtual model endpoints. Rejects any endpoint whose TPR capacity is too small or that exceeds current TPM/RPM usage rates.
5. **Utility Scoring**: For all eligible standard nodes, selects the node that minimizes:
   `Utility = (tier_mismatch, ep.cost_per_million, current_utilization)`
   - Tier mismatch penalty: strong penalty (+5.0) if a high-complexity prompt is sent to a low-cost node, or a low-complexity prompt is sent to an expensive node.
6. **API Dispatch**: Runs LiteLLM's `acompletion` on the chosen endpoint.
7. **Intra-Cluster Failover**: If the API call fails, attempts to failover to other nodes within the same cluster.
8. **Inter-Cluster Failover**: If intra-cluster failover fails, cascades through the fallback clusters.
9. **Usage Tracking**: Invokes `track_request_usage` and updates accumulated metrics and credit spend.

###### Business Logic
Implements cost and resource optimization policies, prioritizing cost savings for simple tasks and high-parameter reasoning for complex coding/math tasks.

###### Database Operations
Tracks sliding-window telemetry by calling ZADD/ZREM operations on Redis.

###### External API Calls
Dispatches completion requests to external APIs (Groq, Cerebras, Together AI) via LiteLLM.

###### AI Model Calls
Invokes local Ollama/Mistral for dynamic PII rewriting tasks.

---

# proxy/app.py

## Purpose
FastAPI web controller exposing endpoints for OpenAI chat completions, telemetry metrics, preference sync, and model fine-tuning.

## Responsibilities
- Configures CORS middleware.
- Exposes OpenAI-compatible `/v1/chat/completions` and `/v1/models` routes.
- Exposes GET/POST endpoints for custom configuration syncing.
- Manages the background training task for DeBERTa model fine-tuning.

## Dependencies
- `fastapi` (Web Framework)
- `pydantic` (Request/Response validation)

## Classes

### `LiteLLMProxyApp`

#### Purpose
Application container encapsulating the configuration parser, proxy router, routes registry, and training states.

#### Constructor
`__init__(self, config_path: str = "config.yaml")`
- Instantiates `ProxyConfig` and `LiteLLMProxyRouter`.
- Checks for local config `finetuned-deberta/config.json` to initialize the training state status (defaults to `"completed"` if local weights exist, else `"idle"`).
- Instantiates the `FastAPI` app.
- Registers API routes.

#### Methods

##### `_register_routes`

###### Purpose
Registers the path endpoints of the application.

###### Endpoints

- **`GET /`**:
  - Redirects users to the deployed Streamlit dashboard console.

- **`GET /health`**:
  - Health check endpoint returning status `"healthy"`.

- **`GET /metrics`**:
  - Returns real-time metrics, active routing rules, and loaded endpoints.

- **`GET /v1/models`**:
  - Lists virtual clusters and physical backend model IDs.

- **`POST /v1/chat/completions`**:
  - OpenAI-compatible completion endpoint. Validates requests and calls `router.execute_chat_completion`.

- **`GET /preference-config`** and **`POST /preference-config`**:
  - Syncs priority preferences list, credit limits, and enablement flags.

- **`POST /preference-config/reset`**:
  - Resets all accumulated spends to `$0.00`.

- **`GET /ui/pii-config`** and **`POST /ui/pii-config`**:
  - Exposes config dashboard sync paths for active labels and policies.

- **`POST /v1/deberta/train`** (and `/ui/train-deberta`):
  - Spawns the background subprocess training thread.

- **`GET /v1/deberta/train/status`**:
  - Returns the training progress state and pulls loss logs from `training_history.json`.

- **`POST /v1/deberta/train/reset`**:
  - Resets status to `"idle"`.

- **`POST /ui/pii-detect`**:
  - Runs raw PII detection using the active model for UI validation.

---

##### `run_training_in_background` (Helper inside `_register_routes`)

###### Purpose
Target function for background threads. Formats the dataset, unloads active inference pipelines to release GPU/RAM memory, saves configuration settings to a temporary JSON file, and spawns the training loop as a subprocess.

###### Parameters
- `req` (`TrainDebertaRequest`): Labeled training samples and hyperparameter settings.

###### Internal Logic
1. Iterates over datasets to build standard start/end character offsets.
2. Loops through `active_guardrails` and sets `_model = None` to unload weights and trigger garbage collection.
3. Writes parameters to `deberta_train_cfg_*.json` in a temp directory.
4. Spawns the subprocess:
   `python deberta_pii_guardrail.py --train <temp_config_path>`
5. Sets the `LITELLM_TRAINING_ACTIVE="1"` env variable to bypass LiteLLM callback setups in the training process.
6. Blocks until execution completes.
7. Calls `reload_model()` on all active pipeline wrappers to load the new weights.
8. Saves new custom label names to `pii_guardrail_config.json`.
9. Sets status to `"completed"`.

###### Side Effects
- Temporarily unloads the active inference model pipeline.
- Spawns a background training subprocess.
- Overwrites the model weight files inside `models/finetuned-deberta/` upon successful training completion.

---

# proxy/templates/index.html

## Purpose
Exposed static UI dashboard built with HTML, CSS (Tailwind CSS), and vanilla JavaScript.

## Responsibilities
- Renders system load gauges, request logs, latency counters, and fallback events.
- Provides forms to update priorities, edit credit limits, select remediation actions, and test queries.
- Connects to the backend via dynamic APIs or Vercel Proxy endpoints.

---

## Architectural Fit

The proxy classes follow a structured, layered OOP architecture:

```
[ main.py ] -> Bootstraps Uvicorn and loads the ASGI App container
      │
      ▼
[ LiteLLMProxyApp ] -> Exposes FastAPI Controllers, REST API routes, and manages background SFT workers
      │
      ├─► [ ProxyConfig ] -> Parses YAML and validates catalog parameters
      │
      └─► [ LiteLLMProxyRouter ] -> Main routing engine. Manages request execution and limits
                │
                ├─► [ LiteLLM Router ] -> Native load balancing & failover retries
                │
                ├─► [ DeBERTaPIIGuardrail ] -> PII detection & remediation
                │
                └─► [ Redis Client ] -> Centralized sliding-window token metrics
```
