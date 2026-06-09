import streamlit as st
import requests
import os
import time
import re
import yaml
import json

# Set minimal, professional industry-appropriate page configuration
st.set_page_config(
    page_title="LiteLLM Gateway Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Minimal CSS (Light Theme / Corporate Dark Accent)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem !important;
    }
    
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 0.95rem;
        color: #6B7280;
        margin-bottom: 1.8rem;
    }
    
    .gateway-url-badge {
        background-color: #F3F4F6;
        color: #374151;
        border: 1px solid #E5E7EB;
        padding: 6px 12px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.25rem;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.75rem;
    }
    
    .metadata-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF;
        font-weight: 600;
    }
    
    .metadata-value {
        font-size: 1.1rem;
        font-weight: 500;
        color: #111827;
    }
    
    .diff-box {
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }
    
    .diff-original {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        color: #991B1B;
    }
    
    .diff-shielded {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        color: #065F46;
    }
    
    .diff-response {
        background-color: #F9FAFB;
        border-left: 4px solid #6B7280;
        color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)

# Port configuration
PROXY_PORT = os.environ.get("PORT", "8005")
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"

# Dynamically parse config.yaml models
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_yaml_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            pass
    return {}

def load_pii_config():
    try:
        r = requests.get(f"{PROXY_URL}/ui/pii-config", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"pii_enabled": False, "pii_action": "MASK"}

# Dynamically parse config.yaml models and default specs
config_data = load_yaml_config()
model_list = config_data.get("model_list", [])

configured_models = []
model_defaults = {}

for item in model_list:
    name = item.get("model_name")
    if name and name not in configured_models:
        configured_models.append(name)
    params = item.get("litellm_params", {})
    sub_model = params.get("model")
    if sub_model and sub_model not in configured_models:
        configured_models.append(sub_model)
        
    tpm_val = params.get("tpm", 50000)
    cost_per_mil = params.get("cost_per_million", 0.05)
    cost_per_k = cost_per_mil / 1000.0
    
    if name and name not in model_defaults:
        model_defaults[name] = {"tpm": tpm_val, "cost": cost_per_k}
    if sub_model and sub_model not in model_defaults:
        model_defaults[sub_model] = {"tpm": tpm_val, "cost": cost_per_k}

if not configured_models:
    configured_models = ["primary-cluster", "backup-cluster", "groq/llama-3.1-8b-instant", "cerebras/llama3.1-8b"]

# Ensure session states exist
if "agent_name" not in st.session_state:
    st.session_state.agent_name = "Credit Assessment Manager"
if "model_priorities" not in st.session_state:
    st.session_state.model_priorities = [configured_models[0]] if configured_models else ["primary-cluster"]
if "tpm_limits" not in st.session_state:
    st.session_state.tpm_limits = {}
if "cost_limits" not in st.session_state:
    st.session_state.cost_limits = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "cost_limit" not in st.session_state:
    st.session_state.cost_limit = 5.00


# ---------------------------------------------------------------------
# SIDEBAR - AGENT DELEGATION
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h3 style='font-weight:600; margin-top:10px; color:#111827;'>Agent Delegation</h3>", unsafe_allow_html=True)
    st.write("Define and delegate context-specific configurations to an autonomous agent.")
    
    agent_options = [
        "Credit Assessment Manager",
        "Customer Support Specialist",
        "Legal Compliance Auditor",
        "Custom Agent..."
    ]
    
    selected_agent = st.selectbox(
        "Active Agent Designation",
        options=agent_options,
        index=agent_options.index(st.session_state.agent_name) if st.session_state.agent_name in agent_options else 0
    )
    
    if selected_agent == "Custom Agent...":
        custom_name = st.text_input("Define Custom Agent Name", value="Risk Analyst Agent")
        st.session_state.agent_name = custom_name
    else:
        st.session_state.agent_name = selected_agent
        
    st.markdown("---")
    st.markdown("<h4 style='font-weight:500; font-size:0.9rem; color:#4B5563;'>Agent Context</h4>", unsafe_allow_html=True)
    
    if st.session_state.agent_name == "Credit Assessment Manager":
        st.caption("Delegated to evaluate financial eligibility, credit ratings, credit card limits, and risk profiles.")
    elif st.session_state.agent_name == "Customer Support Specialist":
        st.caption("Delegated to resolve general client queries.")
    elif st.session_state.agent_name == "Legal Compliance Auditor":
        st.caption("Delegated to audit legal agreements, contract values, and compliance status.")
    else:
        st.caption("Custom agent rules dynamically configured across the current execution path.")

# ---------------------------------------------------------------------
# MAIN INTERFACE TABS (Page 1 vs Page 2)
# ---------------------------------------------------------------------
st.markdown("<div class='main-header'>LiteLLM Gateway Console & Routing Proxy</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Minimalist enterprise routing controls configured for <b>{st.session_state.agent_name}</b></div>", unsafe_allow_html=True)

tab_backend, tab_testing, tab_training = st.tabs([
    "Page 1: Backend Gateway Controls",
    "Page 2: User Testing Interface",
    "Page 3: Model Fine-Tuning"
])

# ---------------------------------------------------------------------
# PAGE 1: BACKEND CONTROLS
# ---------------------------------------------------------------------
with tab_backend:
    col_llm, col_guardrails = st.columns([1, 1], gap="large")
    
    with col_llm:
        st.markdown("<div class='card-title'>LLM Prioritization & Limits</div>", unsafe_allow_html=True)
        st.write("Prioritize execution paths based on TPR/TPM loads. LiteLLM handles fallback routing dynamically.")
        
        # 1. LLM priorities
        order = st.multiselect(
            "Priority Routing Hierarchy",
            options=configured_models,
            default=st.session_state.model_priorities if all(m in configured_models for m in st.session_state.model_priorities) else [configured_models[0]],
            help="LiteLLM Gateway will route requests down this chain on congestion or failure."
        )
        if order:
            st.session_state.model_priorities = order
            
        # 2. Individual Model Limits and Costs
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#374151;'>Individual Model Settings</span>", unsafe_allow_html=True)
        
        for m in st.session_state.model_priorities:
            defaults = model_defaults.get(m, {"tpm": 50000, "cost": 0.05})
            
            # Setup session state key defaults
            if f"tpm_{m}" not in st.session_state:
                st.session_state[f"tpm_{m}"] = defaults["tpm"]
            if f"cost_{m}" not in st.session_state:
                st.session_state[f"cost_{m}"] = defaults["cost"]
                
            st.markdown(f"<div style='font-size:0.85rem; font-weight:500; margin-top:8px; color:#4B5563;'>↳ {m}</div>", unsafe_allow_html=True)
            col_tpm, col_cost = st.columns(2)
            with col_tpm:
                st.session_state.tpm_limits[m] = st.number_input(
                    f"TPM Limit",
                    min_value=1000,
                    max_value=1000000,
                    key=f"tpm_{m}",
                    step=5000
                )
            with col_cost:
                st.session_state.cost_limits[m] = st.number_input(
                    f"Cost/1K Tokens ($)",
                    min_value=0.00001,
                    max_value=100.0,
                    key=f"cost_{m}",
                    format="%.6f",
                    step=0.001
                )
        
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        st.info("ℹ️ LiteLLM load balancer distributes prompt workloads via the dynamic `usage-based-routing` strategy mapped in `config.yaml`.")
        
    with col_guardrails:
        st.markdown("<div class='card-title'>PII Guardrail Controls (DeBERTa-v3)</div>", unsafe_allow_html=True)
        st.write("Apply real-time PII detection and remediation to prompt inputs and model responses.")
        
        pii_cfg = load_pii_config()
        pii_enabled_default = pii_cfg.get("pii_enabled", False)
        pii_action_default = pii_cfg.get("pii_action", "MASK")
        
        pii_enabled = st.toggle(
            "Enable PII Guardrail",
            value=pii_enabled_default,
            help="Scan user prompt inputs and generated model outputs for personally identifiable information (PII)."
        )
        
        actions = ["BLOCK", "MASK", "REWRITE"]
        pii_action = st.selectbox(
            "Default PII Remediation Action",
            options=actions,
            index=actions.index(pii_action_default) if pii_action_default in actions else 1,
            help="Choose the default remediation strategy. Individual overrides can be set below."
        )
        
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#374151;'>Entity-Level Remediation Policies</span>", unsafe_allow_html=True)
        
        pii_policy_default = pii_cfg.get("pii_policy") or {}
        if not isinstance(pii_policy_default, dict):
            pii_policy_default = {}
            
        PII_ENTITIES = {
            "person": "Name",
            "phone number": "Phone Number",
            "social security number": "SSN / Aadhaar",
            "credit card number": "Credit Card",
            "api key": "API Key",
            "email address": "Email Address",
            "address": "Address",
            "bank account number": "Bank Account Number",
            "passport number": "Passport Number",
            "password": "Password"
        }
        
        pii_policy = {}
        with st.expander("Configure Specific PII Types", expanded=True):
            col_ent1, col_ent2 = st.columns(2)
            for idx, (entity_key, entity_label) in enumerate(PII_ENTITIES.items()):
                col = col_ent1 if idx % 2 == 0 else col_ent2
                with col:
                    default_action = pii_policy_default.get(entity_key, pii_action)
                    options = ["BLOCK", "MASK", "REWRITE", "IGNORE"]
                    selected_action = st.selectbox(
                        entity_label,
                        options=options,
                        index=options.index(default_action) if default_action in options else options.index(pii_action),
                        key=f"pii_action_{entity_key}"
                    )
                    pii_policy[entity_key] = selected_action
                    
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        if st.button("Apply PII Policy", type="primary", use_container_width=True):
            payload = {
                "pii_enabled": pii_enabled,
                "pii_action": pii_action,
                "pii_policy": pii_policy
            }
            try:
                r = requests.post(f"{PROXY_URL}/ui/pii-config", json=payload, timeout=5.0)
                if r.status_code == 200:
                    st.success("PII Guardrail configuration updated successfully.")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Failed to save configuration: {r.text}")
            except Exception as e:
                st.error(f"Could not connect to proxy server: {e}")

# ---------------------------------------------------------------------
# PAGE 2: USER TESTING INTERFACE
# ---------------------------------------------------------------------
with tab_testing:
    st.markdown("<div class='card-title'>LLM Unified Gateway Endpoint</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='gateway-url-badge'>POST &nbsp; {PROXY_URL}/v1/chat/completions</div>", unsafe_allow_html=True)
    st.write("")
    
    # User Input Query
    user_query = st.text_area(
        "Enter Query Prompt to Test",
        value="I am Samuel, my phone numbers are +1 213 555-0123 and +91 9876534567 , Aadhaar is 9988-7766-5544, SSN: 111-22-3333. my email is sam@gmail.com,  check which all accounts are linked together?  Also my credit card number is 2345 5432 8765 , check my account balance."
        height=120
    )
    
    if st.button("Submit Request", type="primary"):
        # Trigger Gateway Call
        payload = {
            "model": st.session_state.model_priorities[0] if st.session_state.model_priorities else "oss-chat-fast",
            "messages": [{"role": "user", "content": user_query}],
            "temperature": 0.3,
            "max_tokens": 150
        }
        
        start_time = time.time()
        backend_response = ""
        actual_model = "Mock-Fallback-Node"
        latency = 0.0
        
        try:
            r = requests.post(f"{PROXY_URL}/v1/chat/completions", json=payload, timeout=15)
            latency = time.time() - start_time
            if r.status_code == 200:
                data = r.json()
                backend_response = data["choices"][0]["message"]["content"]
                actual_model = data.get("model", payload["model"])
                guardrailed_query = data.get("guardrailed_query", user_query)
            elif r.status_code == 400:
                try:
                    error_detail = r.json().get("detail", r.text)
                except Exception:
                    error_detail = r.text
                backend_response = f"⚠️ Request Blocked: {error_detail}"
                actual_model = "Blocked (PII Policy)"
                guardrailed_query = None
            else:
                backend_response = f"⚠️ Gateway Error ({r.status_code}): {r.text}"
                guardrailed_query = None
        except Exception as e:
            # Fallback mock response for testing disconnected modes
            latency = 0.045
            backend_response = f"This is a simulated response from the gateway node running in local sandbox mode."
            actual_model = f"{st.session_state.model_priorities[0]} (Local Sandbox Simulation)"
            guardrailed_query = user_query
            
        # Save to stateful variables
        st.session_state.last_result = {
            "query": user_query,
            "response": backend_response,
            "model": actual_model,
            "latency": latency,
            "guardrailed_query": guardrailed_query
        }
        st.session_state.history.append(st.session_state.last_result)
        
    # Render last result if present
    if st.session_state.last_result:
        res = st.session_state.last_result
        
        # Display stacked comparative layout with wide response space
        col_prompts, col_response = st.columns([1.2, 1.8], gap="large")
        
        with col_prompts:
            st.markdown("<span style='font-size:0.8rem; font-weight:600; color:#374151;'>ORIGINAL QUERY</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='diff-box diff-original'>{res['query']}</div>", unsafe_allow_html=True)
            
            # Display guardrailed query below original query
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            st.markdown("<span style='font-size:0.8rem; font-weight:600; color:#374151;'>GUARDRAILED QUERY (SENT TO LLM)</span>", unsafe_allow_html=True)
            
            g_query = res.get("guardrailed_query")
            if g_query is None:
                st.markdown("<div class='diff-box diff-original' style='border-left-color: #EF4444; background-color: #FEF2F2; color: #EF4444; font-style: italic;'>[BLOCKED - NOT SENT TO LLM]</div>", unsafe_allow_html=True)
            elif g_query == res['query']:
                st.markdown(f"<div class='diff-box diff-shielded' style='border-left-color: #9CA3AF; background-color: #F9FAFB; color: #4B5563;'>{g_query}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='diff-box diff-shielded'>{g_query}</div>", unsafe_allow_html=True)
            
        with col_response:
            st.markdown("<span style='font-size:0.8rem; font-weight:600; color:#374151;'>GATEWAY RESPONSE</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='diff-box diff-response' style='min-height:220px;'>{res['response']}</div>", unsafe_allow_html=True)
            
        # Display professional HUD Metrics
        st.markdown("---")
        col_hud1, col_hud2, col_hud3, col_hud4 = st.columns(4)
        
        with col_hud1:
            st.markdown("<div class='metadata-label'>Active Agent</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metadata-value'>{st.session_state.agent_name}</div>", unsafe_allow_html=True)
            
        with col_hud2:
            st.markdown("<div class='metadata-label'>Latency</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metadata-value'>{res['latency']:.3f}s</div>", unsafe_allow_html=True)
            
        with col_hud3:
            st.markdown("<div class='metadata-label'>Target Routing Cluster</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metadata-value'>{res['model']}</div>", unsafe_allow_html=True)
            
        with col_hud4:
            model_tpm = st.session_state.tpm_limits.get(res['model'], 50000)
            model_cost = st.session_state.cost_limits.get(res['model'], 0.05)
            st.markdown("<div class='metadata-label'>Limits & Capacity</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metadata-value'>{model_tpm} TPM / ${model_cost:.5f}</div>", unsafe_allow_html=True)

    # Past queries of the session
    if st.session_state.history:
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:20px 0; border:0; border-top:1px solid #E5E7EB;'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Session Queries History</div>", unsafe_allow_html=True)
        
        for idx, item in enumerate(reversed(st.session_state.history)):
            turn_idx = len(st.session_state.history) - idx
            with st.expander(f"Turn #{turn_idx} &mdash; Model: {item['model']} &mdash; Latency: {item['latency']:.3f}s"):
                col_hist_left, col_hist_right = st.columns([1, 1])
                with col_hist_left:
                    st.markdown("<span style='font-size:0.75rem; font-weight:600; color:#4B5563;'>Original Query</span>", unsafe_allow_html=True)
                    st.info(item["query"])
                    
                    g_query = item.get("guardrailed_query")
                    if g_query is None:
                        st.markdown("<span style='font-size:0.75rem; font-weight:600; color:#DC2626;'>Guardrail Action</span>", unsafe_allow_html=True)
                        st.error("Blocked — Not sent to LLM")
                    elif g_query != item["query"]:
                        st.markdown("<span style='font-size:0.75rem; font-weight:600; color:#D97706;'>Guardrailed Query (Sent to LLM)</span>", unsafe_allow_html=True)
                        st.warning(g_query)
                with col_hist_right:
                    st.markdown("<span style='font-size:0.75rem; font-weight:600; color:#4B5563;'>Gateway Response</span>", unsafe_allow_html=True)
                    st.success(item["response"])


# ---------------------------------------------------------------------
# PAGE 3: MODEL FINE-TUNING
# ---------------------------------------------------------------------
with tab_training:
    st.markdown("<div class='card-title'>DeBERTa-v3 PII Model Fine-Tuning Console</div>", unsafe_allow_html=True)
    st.write("Further train the PII token-classification model locally with custom domain-specific data to improve precision and recall.")
    
    # 1. Fetch current training status from backend
    status_data = {"status": "idle", "progress": "", "error": None}
    try:
        r = requests.get(f"{PROXY_URL}/ui/train-deberta/status", timeout=2.0)
        if r.status_code == 200:
            status_data = r.json()
    except Exception as e:
        st.warning(f"Could not check training status from backend proxy: {e}")
        
    current_status = status_data.get("status", "idle")
    current_progress = status_data.get("progress", "")
    current_error = status_data.get("error")
    
    # 2. Render Status HUD Card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col_hud1, col_hud2 = st.columns([1, 3])
    with col_hud1:
        st.markdown("<span class='metadata-label'>Training Status</span>", unsafe_allow_html=True)
        if current_status == "idle":
            st.markdown("<h3 style='color:#6B7280; margin-top:5px; font-weight:600;'>IDLE</h3>", unsafe_allow_html=True)
        elif current_status == "training":
            st.markdown("<h3 style='color:#3B82F6; margin-top:5px; font-weight:600;'>TRAINING</h3>", unsafe_allow_html=True)
        elif current_status == "completed":
            st.markdown("<h3 style='color:#10B981; margin-top:5px; font-weight:600;'>COMPLETED</h3>", unsafe_allow_html=True)
        elif current_status == "failed":
            st.markdown("<h3 style='color:#EF4444; margin-top:5px; font-weight:600;'>FAILED</h3>", unsafe_allow_html=True)
            
    with col_hud2:
        st.markdown("<span class='metadata-label'>Current Progress / Action</span>", unsafe_allow_html=True)
        if current_status == "idle":
            st.write("Ready to receive training parameters and dataset inputs.")
        elif current_status == "training":
            st.write(f"⏳ **{current_progress}**")
        elif current_status == "completed":
            st.write("🎉 **Model successfully fine-tuned!** The proxy has reloaded the active pipelines to use the new weights.")
        elif current_status == "failed":
            st.write("❌ **Training aborted due to an error.** See details below.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 3. Handle status-specific display states
    if current_status == "training":
        # Poll status in loop using progress indicator
        st.spinner("Fine-tuning model. Please wait...")
        time.sleep(3.0)
        st.rerun()
        
    elif current_status == "completed":
        st.balloons()
        if st.button("Reset Console Status"):
            try:
                requests.post(f"{PROXY_URL}/ui/train-deberta/reset", timeout=2.0)
                st.rerun()
            except Exception:
                pass
                
    elif current_status == "failed":
        st.error("Training Traceback Log:")
        st.code(current_error or "Unknown error occurred.")
        if st.button("Clear Error & Reset"):
            try:
                requests.post(f"{PROXY_URL}/ui/train-deberta/reset", timeout=2.0)
                st.rerun()
            except Exception:
                pass
                
    else: # idle status - render configuration editor
        st.markdown("<div class='card-title'>Configure Hyperparameters</div>", unsafe_allow_html=True)
        col_hp1, col_hp2, col_hp3 = st.columns(3)
        with col_hp1:
            epochs = st.number_input("Epochs", min_value=1, max_value=20, value=3, step=1)
        with col_hp2:
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=5e-5, format="%.6f")
        with col_hp3:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=8, step=1)
            
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Input Training Dataset (JSON Format)</div>", unsafe_allow_html=True)
        st.write("Define text training samples and label offsets for NER Token Classification. Matches canonical labels.")
        
        default_dataset = [
            {
                "text": "Hello, my name is Arthur Pendragon and my email address is arthur@camelot.org.",
                "entities": [
                    {"start": 18, "end": 35, "label": "person"},
                    {"start": 59, "end": 77, "label": "email address"}
                ]
            },
            {
                "text": "Please charge the balance to card number 4111-2222-3333-4444.",
                "entities": [
                    {"start": 41, "end": 60, "label": "credit card number"}
                ]
            },
            {
                "text": "My SSN is 000-12-3456 and I live at 12 Round Table Lane, London.",
                "entities": [
                    {"start": 10, "end": 21, "label": "social security number"},
                    {"start": 36, "end": 63, "label": "address"}
                ]
            }
        ]
        
        dataset_json = st.text_area(
            "Training Dataset JSON",
            value=json.dumps(default_dataset, indent=2),
            height=300
        )
        
        if st.button("Start Fine-Tuning", type="primary", use_container_width=True):
            try:
                parsed_dataset = json.loads(dataset_json)
                if not isinstance(parsed_dataset, list):
                    st.error("Dataset must be a list of training samples.")
                else:
                    # Validate format
                    valid = True
                    for sample in parsed_dataset:
                        if "text" not in sample or "entities" not in sample:
                            st.error("Each sample must contain 'text' and 'entities' properties.")
                            valid = False
                            break
                    if valid:
                        payload = {
                            "dataset": parsed_dataset,
                            "epochs": int(epochs),
                            "learning_rate": float(learning_rate),
                            "batch_size": int(batch_size)
                        }
                        r = requests.post(f"{PROXY_URL}/ui/train-deberta", json=payload, timeout=5.0)
                        if r.status_code == 200:
                            st.success("Training task submitted successfully. Starting background threads...")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Failed to submit training job: {r.text}")
            except json.JSONDecodeError as je:
                st.error(f"Invalid JSON Syntax: {je}")
            except Exception as e:
                st.error(f"Submission failed: {e}")

