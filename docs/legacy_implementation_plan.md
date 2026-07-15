# Implementation Plan: Complexity-Aware Multi-Tier Routing & Cost/Resource Optimization

This plan outlines the design and integration of an enterprise-grade **Complexity-Aware Multi-Tier Routing Engine** into the LiteLLM Proxy server. It classifies incoming prompts dynamically into **Low**, **Medium**, and **High** complexity levels, and routes them to optimal physical LLM backends to maximize response quality while minimizing credit expenditure and system resource load.

---

## User Review Required

> [!IMPORTANT]
> **Dynamic Model Catalog Upgrades & Local Backup Placement**
> To support high-fidelity reasoning for **High Complexity** queries (like coding, algorithm design, or multi-step logic), we will add high-parameter 70B models to our catalog using your existing API keys:
> 1. **Groq**: Add `groq/llama-3.3-70b-versatile` ($0.70/M tokens, 128k context).
> 2. **Together AI**: Add `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` ($0.90/M tokens, 128k context).
>
> **Local Backup Placement**:
> Per your request, the local `ollama/llama3.1` model (cost: $0.00/M tokens) will be placed inside the **`backup-cluster`** (previously under its own local cluster). This allows the backup cluster to dynamically load-balance or failover between premium Together AI and zero-cost local Ollama!

> [!TIP]
> **Backward Compatibility Guard**
> All existing virtual model names (such as `primary-cluster`, `backup-cluster`, and `local-fallback-cluster`) and general fallback chains will remain fully supported. The complexity-aware router will intercept requests to these endpoints and intelligently route them to the most optimal underlying physical endpoint matching the prompt's characteristics.

---

## Proposed Changes

### 1. Configuration & Models

#### [MODIFY] [config.yaml](file:///Users/sanvijain/EY_DataAndAI/LiteLLM_Proxy/config.yaml)
- Add explicit `complexity_tier` metadata properties to existing models.
- Move local Ollama (`ollama/llama3.1`) into `backup-cluster`.
- Add new **High-Complexity** premium endpoints to the cluster lists:
  - Groq Llama 3.3 70B (`groq/llama-3.3-70b-versatile`) under `primary-cluster` (tier: `high`, cost: 0.70, tpr: 131072).
  - Together AI Llama 3.3 70B (`together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo`) under `backup-cluster` (tier: `high`, cost: 0.90, tpr: 131072).

### 2. Configuration Parser

#### [MODIFY] [proxy/config.py](file:///Users/sanvijain/EY_DataAndAI/LiteLLM_Proxy/proxy/config.py)
- Extend `ModelEndpointConfig` Pydantic model with a new field: `complexity_tier: Optional[str] = Field(None, description="Complexity tier for this endpoint: low, medium, or high")`.
- Update `load_config` parser to read `complexity_tier` from `litellm_params`.
- Add an intelligent fallback to default the tier based on pricing and context if not explicitly set in YAML:
  - `cost_per_million >= 0.50` -> `high`
  - `cost_per_million >= 0.04` -> `medium`
  - `cost_per_million < 0.04` -> `low`

### 3. Routing Engine

#### [MODIFY] [proxy/router.py](file:///Users/sanvijain/EY_DataAndAI/LiteLLM_Proxy/proxy/router.py)
- **Prompt Complexity Classifier**: Implement `classify_prompt_complexity(self, messages: List[Dict[str, str]], required_context: int) -> str` to assess prompts based on:
  - **Required Context Length**: Any query requiring >8K context is auto-classified as `high`.
  - **Semantic Indicators**: Scans user input for coding keywords (`python`, `sql`, `refactor`, `debug`), mathematical/logical keywords (`solve`, `prove`, `theorem`, `logic`), or deep analysis tasks (`optimize`, `architecture`, `step-by-step`).
- **Complexity-Aware Selection**: Refactor `execute_chat_completion` to score available endpoints using a multi-objective utility function:
  - **Tier Match**: Strongly rewards endpoints aligning with the classified complexity of the prompt.
  - **Cost Optimization**: Prefers nodes with the lowest `cost_per_million`.
  - **Resource Optimization**: Prefers nodes with the lowest RPM/TPM usage to spread system load.
- Ensure context-window overflows dynamically trigger high-capacity models regardless of semantic complexity.

### 4. Integration Verification

#### [MODIFY] [test_proxy.py](file:///Users/sanvijain/EY_DataAndAI/LiteLLM_Proxy/test_proxy.py)
- Add a new comprehensive unit test: `test_7_complexity_routing`:
  - Submit a conversational greeting ("Hello! How is it going?") and verify it routes to a `low` complexity node (Cerebras or Ollama).
  - Submit a complex python debugging prompt ("Optimize this SQL query and write a python refactoring function...") and verify it routes to a `high` complexity node (Groq 70B or Together 70B, or standard large backup if sandbox).

---

## Verification Plan

### Automated Tests
- Run `python3 test_proxy.py` to ensure all 7 tests pass successfully with **100% success**.

### Manual Sandbox Verification
- Launch the FastAPI server: `python3 main.py`.
- Run mock sandbox queries from the console or Streamlit dashboard on Port `8502`.
- Observe logs in the Telemetry dashboard confirming the dynamic complexity assessments and selected cost-optimized targets.
