# File Documentation: Synthetic Data Package

This document provides detailed class, function, and interface-level documentation for components inside the synthetic data engine.

---

# synthetic_data.py

## Purpose
Implements a diversity-driven synthetic data generation engine to generate balanced training datasets for training PII token classification models.

## Responsibilities
- Generates structured dataset sentences matching specific domain formats (e.g. medical bills, customer tickets).
- Parses regular expressions using `exrex` to generate format-compliant mock inputs (e.g. `PT-[0-9]{5}-[A-Z]{2}`).
- Generates hard negatives (decoy distractor values) using a perturbation algorithm to prevent model over-sensitivity.
- Applies sliding similarity checks using `difflib.SequenceMatcher` to filter out near-duplicate carrier sentences.
- Implements thread pool executors to parallelize requests.
- Validates token start/end character offsets and values in a post-generation pass.

## Dependencies
- `litellm` (LLM completion client)
- `exrex` (Regex text generator)
- `faker` (Mock data provider)
- `pydantic` (Data structures validation)
- `requests` (API requests client)

## Imports
- `import re`, `import json`, `import random`, `import logging`, `import math`, `import time`, `import difflib`
- `from typing import List, Dict, Any, Optional, Tuple, Callable`
- `from concurrent.futures import ThreadPoolExecutor, as_completed`
- `from pydantic import BaseModel, Field`
- `import litellm`, `import requests`
- `from dotenv import load_dotenv`
- `from faker import Faker`
- `import exrex`

## Classes

### `DatasetList`

#### Purpose
Wrapper class that inherits from `list`, adding a `report` attribute to store validation metadata reports.

---

### `CompactEntity`

#### Purpose
Pydantic data model representing a single labeled PII entity span.

#### Attributes
- `l` (`str`): Label name.
- `s` (`int`): Start character offset.
- `e` (`int`): End character offset.
- `v` (`Optional[str]`): Optional verification substring value.

---

### `CompactSample`

#### Purpose
Pydantic model representing a complete training sentence and its entities.

#### Attributes
- `t` (`str`): Text sentence.
- `d` (`str`): Domain context.
- `y` (`str`): Formatting style.
- `e` (`List[CompactEntity]`): List of entity annotations.

---

### `SyntheticDataEngine`

#### Purpose
Main controller encapsulating generators, filters, network managers, and offset validators.

#### Constructor
`__init__(self)`
- Loads `.env` configurations.
- Instantiates a `Faker` instance.
- Initializes metrics parameters (`last_health_check_time`, `is_proxy_healthy`).

#### Methods

##### `is_near_duplicate`

###### Purpose
Checks if a generated sentence is similar to already accepted sentences to maintain dataset diversity.

###### Parameters
- `sentence` (`str`): Candidate sentence.
- `existing` (`List[str]`): Already generated sentences.
- `threshold` (`float`): Similarity threshold (default: `0.85`).

###### Return Value
- `bool`: Returns `True` if similarity exceeds the threshold; otherwise `False`.

###### Internal Logic
1. Replaces `[PII]` strings with empty spaces and normalizes the text.
2. Computes the similarity ratio using `difflib.SequenceMatcher(None, clean_sent, clean_old).ratio()`.

---

##### `_clean_pattern_for_exrex`

###### Purpose
Normalizes regular expressions by removing anchors (`^`, `$`) and word boundaries (`\b`) to make them compatible with `exrex`.

---

##### `generate_regex_value`

###### Purpose
Generates random strings matching custom regular expressions using `exrex`.

###### Parameters
- `pattern` (`str`): Target regular expression.

###### Return Value
- `str`: Mapped string.

###### Exceptions
- Raises `ValueError` if compilation fails.
- Raises `RuntimeError` if it fails to generate a conforming string within 10 attempts.

---

##### `perturb_value`

###### Purpose
Applies character mutations to valid regex strings to create realistic distractors (hard negatives).

###### Parameters
- `val` (`str`): Conforming string value.
- `regex_pattern` (`str`): Target pattern.

###### Return Value
- `str`: Perturbed string that does NOT match the target regex.

###### Internal Logic
Loops up to 30 times, randomly applying one of four mutation strategies:
1. Replaces a character with a random letter/digit.
2. Deletes a character.
3. Inserts a separator character (e.g. `-`, `/`).
4. Replaces a digit with a letter.
Validates each mutation against the target regex, returning the first non-matching string.

---

##### `_generate_fake_value`

###### Purpose
Generates mock values for entity classes using Faker or custom regex patterns.

###### Parameters
- `entity_name` (`str`): Target label name.
- `regex_pattern` (`str`): Pattern constraint.
- `is_hard_negative` (`bool`): If True, perturbs the output.

###### Return Value
- `str`: Generated mock value.

###### Internal Logic
1. If a regex pattern is provided, calls `generate_regex_value` and applies `perturb_value` if `is_hard_negative` is True.
2. Otherwise, maps the entity name to Faker providers:
   - `person` / `name` -> `fake.name()`
   - `email` -> `fake.email()`
   - `phone` -> `fake.phone_number()`
   - `address` -> `fake.address()`
   - `password` -> `fake.password()`
   - `credit card` -> `fake.credit_card_number()`
   - `bank account` -> `fake.bban()`
   - `ssn` -> `fake.ssn()`
   - `api key` -> prefix + `fake.md5()[:16]`
3. If `is_hard_negative` is True and no pattern exists, appends `"_neg"` to the generated Faker value.

---

##### `check_proxy_health`

###### Purpose
Checks if the local proxy gateway is responsive (cached for 30 seconds).

---

##### `_generate_batch`

###### Purpose
Generates a batch of sentences using an LLM.

###### Parameters
- `batch_idx` (`int`): Batch index.
- `entity_name` (`str`): Target label name.
- `regex_pattern` (`str`): Regular expression constraints.
- `domain` (`str`): Target format domain.
- `model` (`str`): Model path to call.
- `proxy_url` (`str`): Target proxy address.
- `is_hard_negative` (`bool`): Generation flag.
- `batch_size` (`int`): Number of sentences to generate.
- `similarity_threshold` (`float`): Near-duplicate threshold.
- `opt_in_cross_label` (`bool`): If True, scans for standard labels as well.

###### Return Value
- `Dict[str, Any]`: Returns `{"samples": validated_samples, "near_dup_count": dup_count}`.

###### Internal Logic
1. Selects a random instruction template from few-shot lists.
2. Constructs the prompt: instructs the model to place the literal placeholder `[PII]` in natural contexts matching the target domain while applying length and formality constraints.
3. Sends completion requests to `/v1/chat/completions` on the proxy. If the proxy is down, falls back to direct API calls via `litellm.completion`.
4. Handles rate limits (`429` status code) with exponential backoff.
5. Extracts and parses the JSON response containing the array of generated sentences.
6. Filters out near-duplicates.
7. Replaces `[PII]` placeholders with generated mock values (calling `_generate_fake_value`).
8. Calculates start/end character offsets for the inserted values.
9. If `opt_in_cross_label` is True, scans the generated sentences using standard regex patterns to label other PII categories.
10. Resolves overlaps by prioritizing the primary target label over other matches.

---

##### `generate_dataset`

###### Purpose
Coordinates the overall parallel generation, recovery loops, and validation checks.

###### Parameters
- `num_samples` (`int`): Target dataset size.
- `target_labels` (`List[str]`): Monitored entities list.
- `model` (`str`): Model to use.
- `progress_callback` (`Optional[Callable]`): Progress reporting callback.
- `synthesis_inputs` (`Optional[List[Dict[str, Any]]]`): Custom label settings (regex and formats).
- `batch_size` (`int`): Batch size (default: `15`).
- `similarity_threshold` (`float`): Similarity threshold (default: `0.85`).
- `hard_negative_ratio` (`float`): Distractor ratio (default: `0.12`).
- `max_workers` (`int`): Thread count limits (default: `3`).
- `domain_pool` (`Optional[List[str]]`): Lists of domain formats.
- `opt_in_cross_label` (`bool`): Label scanning toggle.

###### Return Value
- `DatasetList`: Evaluated samples list with validation metadata report.

###### Internal Logic
1. Computes layout configurations: sets 10% of samples as neutral sentences containing no PII, splits the remainder among the target labels, and applies the `hard_negative_ratio` to set positive vs negative targets.
2. Groups tasks by entity type to prevent context-switching in the LLM.
3. Spawns tasks in parallel using a `ThreadPoolExecutor` and reports status via the progress callback.
4. **Recovery Loop**: If thread execution fails or some samples are dropped, runs a sequential recovery loop to generate the remaining samples.
5. **Validation Pass**: Validates every generated sample:
   - Confirms start/end offsets match the annotated value: `text[start:end] == value`.
   - Validates generated values against custom regex patterns.
   - Drops any invalid samples.
6. Builds a validation report mapping: valid count, dropped count, near-duplicate rate, and domain distributions.
7. Returns a `DatasetList` containing the validated samples.

---

# run_synthetic_engine.py

## Purpose
Command-line interface (CLI) to configure and run the synthetic data engine.

## Functions

### `print_progress`

#### Purpose
Renders a live progress bar and statistics (attempts, successes, healed spans) in the terminal.

---

### `main`

#### Purpose
Parses CLI arguments (`--num-samples`, `--model`, `--output`, `--labels`), instantiates the synthetic data engine, runs generation, saves output to a JSON file, and prints summary metrics.

---

## Architectural Fit

The synthetic data engine runs independently from the proxy completions pipeline, acting as a data provider for model training.

```
[ Streamlit Console / CLI ] ──► [ SyntheticDataEngine ]
                                        │
                                        ▼ (Parallel Thread Pool)
                                [ generate_batch ]
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼ (API requests)                        ▼ (regex/Faker)
          [ litellm/FastAPI ]                     [ generate_fake_value ]
           (generate sentences                     (insert mock values)
            with [PII] carrier)                             │
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                            [ Duplicate/Offset Audit ]
                                        │
                                        ▼
                             [ Saved JSON Dataset ]
                                        │
                                        ▼
                           [ PII Model Fine-Tuning ]
```
