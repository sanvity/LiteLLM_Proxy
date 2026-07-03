# File Documentation: Verification Test Suite

This document details the test suites used to verify configuration parsing, PII masking, routing fallbacks, synthetic generation, and fine-tuning pipelines.

---

# test_proxy.py

## Purpose
Implements a comprehensive suite of unit and integration tests to verify the correctness of the LiteLLM Proxy.

## Responsibilities
- Verifies YAML configuration parsing and endpoint validation.
- Validates the tiktoken token estimation and context estimation logic.
- Verifies Tokens Per Request (TPR) context window filtering and model escalation rules.
- Starts a background FastAPI web server to test OpenAI completions, models list, and metrics endpoints.
- Verifies that queries are routed to the correct model tier based on prompt complexity.
- Validates credit limit tracking and failovers.
- Verifies PII masking, blocking, and dynamic rewriting capabilities.
- Tests the out-of-process training background tasks and custom PII label additions.
- Confirms PII config persistence.

## Dependencies
- `unittest` (Testing framework)
- `requests` (API request validation client)
- `uvicorn` (Spawns testing server instance)
- `asyncio` (Runs async completion tasks)

## Imports
- `import os`, `import time`, `import requests`, `import asyncio`, `import unittest`, `import threading`, `import logging`: Core utilities.
- `from dotenv import load_dotenv`: Environment configurations loader.
- `from proxy import ProxyConfig, LiteLLMProxyRouter, LiteLLMProxyApp`: Import target components under test.

## Classes

### `TestLiteLLMProxy`

#### Purpose
Main test suite class.

#### Methods

##### `setUpClass`

###### Purpose
Initializes test configurations. Deletes any existing `finetuned-deberta` directory and `pii_guardrail_config.json` config file to guarantee a clean slate.

---

##### `test_1_config_loading`

###### Purpose
Verifies configuration details: asserts that endpoints are loaded, fallbacks are populated, and the routing strategy is set to `"usage-based-routing"`.

---

##### `test_2_token_estimation`

###### Purpose
Verifies the token estimation logic for single strings and message arrays.

---

##### `test_3_tpr_context_filtering`

###### Purpose
Verifies TPR context window filtering:
1. Submitting a short prompt should route to primary Llama-3.1-8B (Cerebras or Groq).
2. Submitting an extremely long prompt (>130K tokens) should automatically escalate to high-capacity backup nodes (Together AI Llama 3.3 70B).

---

##### `test_4_mock_sandbox_completion`

###### Purpose
Verifies that completions run correctly in mock sandbox mode.

---

##### `test_5_api_microservice_integration`

###### Purpose
Starts the FastAPI microservice on port 8090 in a background daemon thread, sends requests via HTTP, and asserts that the `/health`, `/v1/models`, `/v1/chat/completions` (mock sandbox), and `/metrics` endpoints function correctly.

---

##### `test_6_complexity_routing`

###### Purpose
Verifies complexity classification and tier routing matching policies:
1. Low complexity greetings route to `cerebras/llama3.1-8b` (Tier: low).
2. Medium complexity summarizations route to `groq/llama-3.1-8b-instant` (Tier: medium).
3. High complexity coding requests route to `groq/llama-3.3-70b-versatile` (Tier: high).

---

##### `test_7_priority_preference_routing`

###### Purpose
Verifies custom priority routing and budget failovers:
1. Configures a preference order: `[groq/llama-3.1-8b-instant, cerebras/llama3.1-8b]`.
2. Sets a tiny credit limit (`$0.000001`) for the first model and a normal limit (`$0.05`) for the second.
3. Submitting the first request should route to the first priority model.
4. Asserts that the first model's credit limit has been exceeded, and verifies that the second request automatically fails over to the second priority model.
5. Resets the spend counters and verifies that routing goes back to the first model.

---

##### `test_8_pii_guardrail`

###### Purpose
Verifies PII masking and blocking capabilities:
1. With action set to `"MASK"`, verifies that SSN strings are replaced with a `[SOCIAL_SECURITY_NUMBER]` placeholder.
2. With action set to `"BLOCK"`, verifies that the request is blocked and raises a `ValueError`.

---

##### `test_9_deberta_model_training`

###### Purpose
Verifies the fine-tuning endpoints: submits a single-epoch training request, polls the status API until completed, and verifies that model weights are written to the target folder.

---

##### `test_x_pii_config_persistence`

###### Purpose
Verifies that updating the PII configuration via API writes the settings to disk, and that a new router instance successfully reads these saved settings on startup.

---

##### `test_z_custom_class_pipeline`

###### Purpose
Tests the end-to-end custom label training and validation pipeline:
1. Asserts that `membership card` is not in the active model labels.
2. Triggers training with a custom class labeled `Membership Card`.
3. Polls status until completed, and asserts that `membership card` is now an active label.
4. Updates the policy to `"MASK"` and verifies that `MEM-98765-XYZ` in the prompt is masked to `[MEMBERSHIP_CARD]`.
5. Updates the policy to `"BLOCK"` and verifies that the query is blocked.

---

# test_synthetic_engine.py

## Purpose
Tests the helper methods and validation routines of the synthetic data generation engine.

## Classes

### `TestSyntheticDataEngine`

#### Purpose
Main test suite class.

#### Methods

##### `test_regex_generation_conformance`

###### Purpose
Verifies that values generated from regex patterns conform to the regular expression.

---

##### `test_hard_negatives_perturbation`

###### Purpose
Asserts that perturbed values do NOT conform to the target regex pattern.

---

##### `test_near_duplicate_filter`

###### Purpose
Verifies that carrier sentences with high similarity (>80%) are flagged as near-duplicates, while dissimilar sentences are allowed.

---

##### `test_overlap_resolution_priority`

###### Purpose
Verifies that overlapping entity spans are correctly resolved, with the primary target label taking precedence over secondary matches.
