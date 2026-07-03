# File Documentation: Streamlit UI Package

This document details the configuration console dashboard implemented in Streamlit.

---

# app.py (Root Directory) & streamlit_app/app.py

## Purpose
Provides an interactive management dashboard to configure the API gateway proxy. It enables live testing of prompts, configuration of PII rules, and supervised fine-tuning of the DeBERTa model.

## Responsibilities
- Implements the **Agent Delegation** control interface.
- Syncs settings (enabled status, policies, custom labels) with the FastAPI backend.
- Renders a drag-and-drop sortable interface for LLM priority routing using `streamlit_sortables`.
- Evaluates real-time metrics (latency, active models, token limits) and displays historical session query lists.
- Hosts training dataset configurations: compiles custom training sets using the synthetic data engine and submits fine-tuning requests.
- Polls training progress status, plotting real-time loss reduction charts.

## Dependencies
- `streamlit` (UI App framework)
- `requests` (API requests client)
- `pyyaml` (YAML parser)
- `streamlit-sortables` (Sortable UI drag-and-drop widgets)
- `synthetic_data` (Data generation backend engine)

## Imports
- `import streamlit as st`: Streamlit application engine.
- `import requests`: For sending config settings and completion requests to the FastAPI backend.
- `import os`, `import time`, `import re`, `import html`, `import json`, `import yaml`: Basic file, timing, string, and structure formatting operations.
- `from synthetic_data import SyntheticDataEngine`: Invokes generation processes directly from the UI thread.
- `from dotenv import load_dotenv`: Loads environment configurations.

## Functions

### `parse_tagged_text`

#### Purpose
Parses XML/HTML-like tagged text strings to extract text values and calculate entity start/end offset character indices.

#### Parameters
- `tagged_text` (`str`): Tagged text string input (e.g. `"Hello, my name is <person>Arthur</person>."`).

#### Return Value
- `dict`: Returns `{"text": clean_text, "entities": entities}`.
  - `clean_text` (`str`): Clean string with tags removed.
  - `entities` (`List[dict]`): Labeled entity offsets. Each entry has keys `start`, `end`, and `label`.

#### Internal Logic
1. Defines the regex pattern: `r'<([a-zA-Z0-9_\- ]+?)>(.*?)</\1>'`.
2. Loops through matches:
   - Appends text before the match to `clean_text`.
   - Records the start character index of the entity.
   - Appends the entity value to `clean_text` and records the end index.
   - Appends the entity offset dictionary to the entities list.
3. Appends the remaining text to `clean_text`.

#### Example Usage
```python
res = parse_tagged_text("Hello, my name is <person>Arthur</person>.")
# Returns:
# {
#   "text": "Hello, my name is Arthur.",
#   "entities": [{"start": 18, "end": 24, "label": "person"}]
# }
```

---

### `load_yaml_config`

#### Purpose
Reads the `config.yaml` file to determine the logical virtual model names and physical endpoints loaded by the proxy.

#### Parameters
None

#### Return Value
- `dict`: Parsed YAML data structures.

---

### `check_backend_health`

#### Purpose
Sends a GET request to the proxy backend `/health` endpoint to verify connectivity. Cached for 10 seconds.

#### Parameters
- `url` (`str`): Target proxy base URL.

#### Return Value
- `bool`: Returns `True` if connected and responsive; otherwise `False`.

---

### `fetch_pii_config`

#### Purpose
Fetches the active PII configuration settings from the backend. Cached for 10 seconds.

#### Parameters
- `url` (`str`): Backend URL.

#### Return Value
- `dict`: Active configuration dictionary.

---

### `fetch_training_status`

#### Purpose
Fetches the active training state and history of loss logs. Cached for 5 seconds.

#### Parameters
- `url` (`str`): Backend URL.

#### Return Value
- `dict`: Mapped progress logs.

---

## User Interface Layout & Controls

The dashboard is structured into three main tabs:

### 1. Tab 1: Backend Gateway Controls
- **LLM Prioritization & Limits**:
  - Uses `streamlit_sortables.sort_items` to present a drag-and-drop interface for ordering models in the active preference list.
  - Dynamically renders text inputs for each active model, allowing admins to adjust TPM rate limits and token unit costs.
- **PII Guardrail Controls (DeBERTa-v3)**:
  - Renders a toggle switch to enable/disable the PII Guardrail.
  - Exposes dropdown selectors to assign remediation actions (MASK, BLOCK, REWRITE, IGNORE) for each supported PII entity type.
  - Provides an "Apply PII Policy" button to save these configurations to the backend.

### 2. Tab 2: User Testing Interface
- Exposes a query test playground panel.
- Submitting a prompt sends a POST request to `/v1/chat/completions` on the proxy backend.
- Displays a comparative side-by-side view showing the original query, the query sent to the LLM (after pre-call guardrail processing), and the generated model response (after post-call guardrail processing).
- Renders metrics gauges showing active latency, selected models, and limits.
- Renders an expander list displaying past session history.

### 3. Tab 3: Model Fine-Tuning
- Renders a status card showing the current training state (IDLE, TRAINING, COMPLETED, FAILED).
- **If training is active**:
  - Automatically re-runs `st.rerun()` every 3 seconds to poll for progress updates.
  - Displays progress strings and plots a line chart showing the real-time loss reduction curve.
- **If training is idle**:
  - Renders a form to add/remove custom PII definitions (format, label, regex pattern).
  - Provides a "Synthesize Dataset" button to invoke the `SyntheticDataEngine` and generate balanced training samples.
  - Displays the generated dataset as an editable JSON text area.
  - Renders hyperparameters configuration sliders (epochs, learning rate, batch size, and Optuna options).
  - Provides a "Start Fine-Tuning" button to submit the training dataset and parameters to the backend.

---

## Architectural Fit

The Streamlit UI acts as the human-in-the-loop dashboard console, communicating with the core microservice via API endpoints.

```
       [ Streamlit UI Dashboard ]
         /           |          \
        /            |           \
 (GET/POST config)   |   (POST chat completion)
      /              |             \
     ▼               ▼              ▼
[ /ui/pii-config ] [ /metrics ] [ /v1/chat/completions ]
     \               |              /
      ▼              ▼             ▼
       [ FastAPI Proxy Backend Server ]
```
