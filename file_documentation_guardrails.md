# File Documentation: Guardrails Package

This document provides detailed class, function, and interface-level documentation for components inside the `guardrails` folder and the model training baselines.

---

# guardrails/deberta_pii_guardrail.py

## Purpose
Implements a production-grade, local Token Classification PII Guardrail using a DeBERTa-v3 model. It hooks into the LiteLLM lifecycle as a `CustomGuardrail` and provides supervised fine-tuning loops.

## Responsibilities
- Performs Named Entity Recognition (NER) token classification on prompt inputs and responses.
- Automatically maps model-specific classifications to canonical labels (e.g. `B-FIRSTNAME` -> `person`).
- Implements remediation policies: MASK (redact content with placeholders), BLOCK (suppress request with a 400 error), and REWRITE (call a local LLM to remove PII text).
- Provides `train_deberta_model()` to train/fine-tune the token classification model on custom domain datasets.
- Manages memory cleanups and processes weights reloads to support seamless model updates.

## Dependencies
- `transformers` (pipeline, AutoModelForTokenClassification, Trainer, TrainingArguments)
- `torch` (gradient computations)
- `litellm` (custom guardrail base class)
- `httpx` / `requests` (Ollama connection client)

## Imports
- `from dataclasses import dataclass`: Standard dataclass wrapper for config configurations.
- `from typing import Dict, List, Optional, Union, Any, Tuple`: Type hints.
- `import asyncio`: Asynchronous event processing.
- `from litellm.integrations.custom_guardrail import CustomGuardrail`: LiteLLM integration wrapper.
- `from litellm._logging import verbose_proxy_logger`: LiteLLM internal logger.

## Classes

### `_Config`

#### Purpose
Read-only dataclass holding PII policies, default actions, thresholds, rewriting models, and label mappings.

#### Constructor
Declared as standard dataclass.

#### Attributes
- `policy` (`Dict[str, str]`): Map of PII label to remediation action (e.g. `{"person": "MASK", "email address": "BLOCK"}`).
- `default_action` (`Optional[str]`): Default action if label is not listed in the policy.
- `threshold` (`float`): Minimum classification probability score threshold (default: `0.4`).
- `rewrite_model` (`str`): Target model for rewriting (default: `"mistral"`).
- `apply_to` (`str`): Target Hook (`"input"`, `"output"`, or `"both"`).
- `labels` (`List[str]`): Monitored labels.

#### Methods
- `action_for(self, label: str) -> Optional[str]`: Resolves the target policy action for a given entity label.

---

### `DeBERTaPIIGuardrail`

#### Purpose
Integrates local DeBERTa PII shielding with the LiteLLM middleware pipeline.

#### Constructor
`__init__(self, **kwargs)`
- Inherits from `CustomGuardrail`.
- Sets local pipeline reference `self._model = None`.
- Registers itself in global callback registry tracker `active_guardrails`.

#### Methods

##### `reload_model`

###### Purpose
Clears the pipeline cache to force reloading model weights on the next request.

---

##### `should_run_guardrail`

###### Purpose
Determines whether this guardrail should intercept current event types.

---

##### `model` (Property)

###### Purpose
Thread-safe getter that initializes the transformers token classification pipeline from local path or base model on demand.

###### Return Value
- `transformers.Pipeline`: Pipeline instance.

---

##### `_get_config`

###### Purpose
Extracts PII parameters from incoming LiteLLM metadata payloads.

---

##### `detect`

###### Purpose
Runs inference on prompt text using the token classification model, filters predictions below the threshold, and sorts them.

###### Parameters
- `text` (`str`): Target prompt text.
- `cfg` (`_Config`): Active configurations.

###### Return Value
- `List[Dict[str, Any]]`: Sorted predictions. Each entry has: `label`, `start` offset, `end` offset, `score`, and matching `text` string.

###### Internal Logic
1. Feeds text into `self.model(text)`.
2. Loops through predicted entities:
   - Skips predictions with `score < cfg.threshold`.
   - Canonicalizes model labels using `canonicalize_label`.
   - If mapped label is monitored, appends start, end, score, and word.
3. Sorts prediction objects in descending order of start indexes.

---

##### `apply_mask`

###### Purpose
Replaces PII spans with label tags.

###### Parameters
- `text` (`str`): Target text.
- `entities` (`List[Dict[str, Any]]`): Detected PII spans.
- `cfg` (`_Config`): Policy configs.

###### Return Value
- `str`: Masked text (e.g. "Hi, my name is [PERSON]").

---

##### `apply_rewrite`

###### Purpose
Sends prompts to Ollama to rewrite text without PII.

###### Parameters
- `text` (`str`): Original text.
- `entities` (`List[Dict[str, Any]]`): Identified spans.
- `cfg` (`_Config`): Configs.

###### Return Value
- `str`: Rewritten text. Falls back to masking if the Ollama endpoint is down.

###### Internal Logic
1. Checks for entities configured with the `REWRITE` action.
2. Constructs the prompt instructing Ollama to replace target labels with generic placeholders (e.g. 'the user') and return ONLY the rewritten text.
3. Sends POST request to `http://localhost:11434/api/generate` with model `cfg.rewrite_model`.
4. Parses JSON and returns response. Falls back to masking on connection failure.

---

##### `check_block`

###### Purpose
Audits detected entities against BLOCK policies.

###### Parameters
- `entities` (`List[Dict[str, Any]]`): Detected entities list.
- `cfg` (`_Config`): Active configs.

###### Return Value
- `Optional[str]`: Returns a violation description string if a block condition is triggered; otherwise `None`.

---

##### `process`

###### Purpose
The main execution pipeline for PII detection, blocking, masking, and rewriting.

###### Parameters
- `text` (`str`): Input text.
- `cfg` (`_Config`): Configurations.

###### Return Value
- `Tuple[str, bool, Optional[str]]`: Returns `(remediated_text, was_modified, block_reason)`.

###### Internal Logic
1. Invokes `self.detect` to identify entity spans.
2. Checks for blocking conditions using `self.check_block`.
3. Splits entities into masking and rewriting groups.
4. Executes remediation:
   - If both MASK and REWRITE entities exist, calls `_get_masked_and_adjusted` to redact masked entities first, then passes the adjusted text to the rewriting pipeline.
   - If only MASK entities exist, calls `self.apply_mask`.
   - If only REWRITE entities exist, calls `self.apply_rewrite`.
5. Returns finalized text, update status flag, and validation status.

---

##### `async_pre_call_hook`

###### Purpose
Intercepts incoming prompt payloads pre-call.

###### Parameters
- `data` (`dict`): LiteLLM prompt payloads.

###### Return Value
- `dict`: Updated payloads.

###### Exceptions
- Raises `litellm.exceptions.BadRequestError` if a PII BLOCK policy is violated.

---

##### `async_post_call_success_hook`

###### Purpose
Intercepts model completions outputs post-call.

###### Parameters
- `response` (`Any`): LiteLLM completion output.

###### Return Value
- `Any`: Sanitized response.

---

## Functions

### `_get_masked_and_adjusted`

#### Purpose
Helper function that handles overlapping MASK and REWRITE entities. It masks target spans first and calculates adjusted offsets for the remaining REWRITE spans.

#### Parameters
- `text` (`str`): Original prompt text.
- `mask_ents` (`list`): Spans targeted for masking.
- `rewrite_ents` (`list`): Spans targeted for rewriting.

#### Return Value
- `Tuple[str, list]`: Returns `(partially_masked_text, adjusted_rewrite_spans)`.

---

### `get_custom_labels_set`

#### Purpose
Reads the fine-tuned model's `config.json` to identify newly added custom PII labels.

#### Return Value
- `set`: Mapped custom label tokens.

#### Internal Logic
1. Locates `config.json` inside the fine-tuned model directory.
2. Reads properties inside `id2label`.
3. Identifies custom indices (index >= 111).
4. Cleans prefix strings (e.g. `B-` / `I-`) and caches results using file modification timestamps.

---

### `canonicalize_label`

#### Purpose
Normalizes model classification tags to standard categories (e.g. mapping `B-FIRSTNAME` and `I-LASTNAME` to `person`).

#### Parameters
- `model_label` (`str`): Raw classification label.

#### Return Value
- `str`: Canonical label string.

---

### `train_deberta_model`

#### Purpose
Performs supervised fine-tuning of the DeBERTa model using custom datasets and hyperparameters.

#### Parameters
- `dataset` (`List[Dict[str, Any]]`): Labeled training dataset list.
- `output_dir` (`str`): Target path directory.
- `epochs` (`int`): Epoches configuration (default: `3`).
- `learning_rate` (`float`): Optimizer learning rate (default: `5e-5`).
- `batch_size` (`int`): Batch size (default: `8`).
- `use_optuna` (`bool`): Toggle hyperparameter search (default: `False`).
- `optuna_trials` (`int`): Optuna iteration trials.

#### Internal Logic / Algorithm
1. Unloads active inference model instances from memory to avoid GPU/RAM overflow.
2. Configures label lists. If the dataset contains labels not in the base model vocabulary (indices >= 111), adds them to `id2label` and `label2id` mappings.
3. Defines `model_init_fn` to load pre-trained weights and dynamically resize the classification layer.
4. Tokenizes dataset: calculates char offset mappings to align token predictions to characters, setting label IDs to `-100` for padded tokens.
5. Computes token class weights to balance the loss, downweighting the background 'O' class to prevent bias.
6. Defines `WeightedTrainer` to compute custom weighted cross-entropy loss.
7. If `use_optuna` is True:
   - Configures a hyperparameter search space (epochs, batch size, and learning rate).
   - Runs `tuning_trainer.hyperparameter_search` using Optuna.
   - Applies the best hyperparameters to the final training run.
8. Scans the directory for existing checkpoints and resumes training from the latest checkpoint if found.
9. Executes `trainer.train()`.
10. Releases the live inference pipeline's file lock.
11. Saves weights to a temporary folder and atomically replaces the target output directory `models/finetuned-deberta/` to prevent directory corruption.

---

# train_pii_deberta.py

## Purpose
Baseline baseline script for supervised fine-tuning of the DeBERTa PII Token Classification model.

## Responsibilities
- Implements class-weighted Cross-Entropy loss inside `WeightedPIITrainer`.
- Sets up model initialization hooks with label mismatch protection.
- Demonstrates Optuna hyperparameter tuning workflows.
- Implements checkpoint auto-detection to resume training after crashes.

---

## Architectural Fit

The guardrails components integrate directly into the LiteLLM Proxy pipeline.

```
                  [ incoming prompt ]
                           │
                           ▼
          [ proxy/router: completions endpoint ]
                           │
                           ▼
         [ DeBERTaPIIGuardrail.async_pre_call_hook ]
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
    [ MASK ]            [ BLOCK ]        [ REWRITE ]
  redact spans        raise 400 error    call Ollama
        │                  │                  │
        ▼                  ▼                  ▼
[ sanitized prompt ]  [ abort req ]      [ rewritten prompt ]
        │                                     │
        └──────────────────┬──────────────────┘
                           │
                           ▼
                 [ execute_chat_completion ]
```
