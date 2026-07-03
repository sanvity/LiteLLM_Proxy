# Developer Technical Documentation Index

Welcome to the technical documentation for the **LiteLLM Load-Balancing & Routing Proxy** system. This documentation is organized into modular files covering different aspects of the codebase.

---

## 🗺️ Documentation Directory

### 1. [System Architecture Overview](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/docs/architecture.md)
Provides a high-level view of system components, request execution data flows, and model training sequences. Contains system block diagrams and Mermaid sequence flows.

### 2. [Folder Directory Documentation](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/docs/folder_documentation.md)
Explains the purpose, reason for existence, interactions, and execution flow for each directory in the codebase (`config/`, `frontend/`, `guardrails/`, `models/`, `proxy/`, `streamlit_app/`).

### 3. [Proxy Package Code Documentation](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/docs/file_documentation_proxy.md)
Detailed class and function-level documentation for:
- `main.py` (server startup entry point)
- `proxy/__init__.py` (package exports)
- `proxy/config.py` (YAML parser and Pydantic schemas)
- `proxy/router.py` (load balancing, complexity routing, and context estimation)
- `proxy/app.py` (FastAPI controller exposing OpenAI endpoints and SFT API tasks)
- `proxy/templates/index.html` (the static HTML/JS client dashboard)

### 4. [PII Guardrail & SFT Code Documentation](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/docs/file_documentation_guardrails.md)
Detailed code documentation for PII classification and baseline training scripts:
- `guardrails/deberta_pii_guardrail.py` (DeBERTa classification pipeline, MASK/BLOCK/REWRITE remediation, and out-of-process fine-tuning loop)
- `train_pii_deberta.py` (baseline training script with Optuna hyperparameter optimization)

### 5. [Synthetic Data Generation Code Documentation](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/docs/file_documentation_synthesis.md)
Detailed code documentation for the synthetic generation pipeline:
- `synthetic_data.py` (Faker/exrex values generator, hard negative distractors, similarity filters, and parallel executors)
- `run_synthetic_engine.py` (command-line CLI runner wrapper)

### 6. [Streamlit Management Console UI Documentation](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/docs/file_documentation_ui.md)
Detailed documentation for the Streamlit dashboard components:
- `app.py` (root management interface)
- `streamlit_app/app.py` (environment-adapted console for cloud deployments)

### 7. [Verification Test Suite Documentation](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/docs/file_documentation_tests.md)
Detailed documentation for the verification test suite:
- `test_proxy.py` (completions, complexity routing, preference limits, and PII guardrail testing)
- `test_synthetic_engine.py` (regex generator, distractor perturbations, and duplicate checks verification)

### 8. [System Features Reference Guide](file:///c:/Users/HP/OneDrive/Desktop/litellm%20deployed%20july/litellm%20deployed%20final/LiteLLM%20Deployed/docs/feature_documentation.md)
Functional reference explaining business requirements, implementation details, inputs, outputs, and constraints for each system feature:
- Complexity-Aware Multi-Tier Routing
- Tokens Per Request (TPR) Context Escalation
- Real-Time PII Guardrail Shield (Masking, Rewriting, Blocking)
- Priority-Based Preference Routing & Credit Budgets
- Asynchronous Supervised Fine-Tuning API
- Bias-Preventing Synthetic Dataset Generation
