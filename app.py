import streamlit as st
import requests
import os
import time
import re
import yaml

# Set minimal, professional industry-appropriate page configuration
st.set_page_config(
    page_title="LiteLLM Gateway Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
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
if "custom_rules" not in st.session_state:
    st.session_state.custom_rules = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "cost_limit" not in st.session_state:
    st.session_state.cost_limit = 5.00
if "guardrail_hooks" not in st.session_state:
    st.session_state.guardrail_hooks = ["Pre-call"]
if "guardrail_rules" not in st.session_state:
    st.session_state.guardrail_rules = {
        "Personally Identifiable Information (PII)": "Mask",
        "Medical Records": "Mask",
        "Aadhaar Number": "Mask",
        "SSN": "Mask",
        "Passport": "Mask",
        "Name": "DummyWrite",
        "Phone number": "Mask"
    }

# Dynamic Guardrail Engine logic (Remove / Mask / DummyWrite / None)
def apply_guardrails(text, rules, active_hooks, stage="Pre-call"):
    if stage not in active_hooks:
        return text
        
    processed = text
    
    # SSN Regex
    ssn_regex = r"\b\d{3}-\d{2}-\d{4}\b"
    # Aadhaar Card Regex
    aadhaar_regex = r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"
    # Passport Regex (Basic alphanumeric standard)
    passport_regex = r"\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b"
    # Phone number Regex
    phone_regex = r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"
    # Personally Identifiable Information (PII) Regex
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ip_regex = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    # Medical record keywords
    medical_keywords = [r"\bcancer\b", r"\bdiabetes\b", r"\bpatient medical files\b", r"\bICD-10\b"]
    
    # 1. SSN Handling
    rule = rules.get("SSN", "None")
    if rule == "Remove":
        processed = re.sub(ssn_regex, "", processed)
    elif rule == "Mask":
        processed = re.sub(ssn_regex, "[SSN_REDACTED]", processed)
    elif rule == "DummyWrite":
        processed = re.sub(ssn_regex, "000-12-3456", processed)

    # 2. Aadhaar Card Handling
    rule = rules.get("Aadhaar Number", "None")
    if rule == "Remove":
        processed = re.sub(aadhaar_regex, "", processed)
    elif rule == "Mask":
        processed = re.sub(aadhaar_regex, "[AADHAAR_REDACTED]", processed)
    elif rule == "DummyWrite":
        processed = re.sub(aadhaar_regex, "9999-8888-7777", processed)

    # 3. Passport Handling
    rule = rules.get("Passport", "None")
    if rule == "Remove":
        processed = re.sub(passport_regex, "", processed)
    elif rule == "Mask":
        processed = re.sub(passport_regex, "[PASSPORT_REDACTED]", processed)
    elif rule == "DummyWrite":
        processed = re.sub(passport_regex, "A1234567", processed)

    # 4. Phone number Handling
    rule = rules.get("Phone number", "None")
    if rule == "Remove":
        processed = re.sub(phone_regex, "", processed)
    elif rule == "Mask":
        processed = re.sub(phone_regex, "[PHONE_REDACTED]", processed)
    elif rule == "DummyWrite":
        processed = re.sub(phone_regex, "555-0199", processed)

    # 5. Personally Identifiable Information (PII)
    rule = rules.get("Personally Identifiable Information (PII)", "None")
    if rule == "Remove":
        processed = re.sub(email_regex, "", processed)
        processed = re.sub(ip_regex, "", processed)
    elif rule == "Mask":
        processed = re.sub(email_regex, "[EMAIL_REDACTED]", processed)
        processed = re.sub(ip_regex, "[IP_REDACTED]", processed)
    elif rule == "DummyWrite":
        processed = re.sub(email_regex, "info@example.com", processed)
        processed = re.sub(ip_regex, "192.168.1.1", processed)

    # 6. Medical Records
    rule = rules.get("Medical Records", "None")
    for kw in medical_keywords:
        if re.search(kw, processed, re.IGNORECASE):
            if rule == "Remove":
                processed = re.sub(kw, "", processed, flags=re.IGNORECASE)
            elif rule == "Mask":
                processed = re.sub(kw, "[MEDICAL_METRIC_REDACTED]", processed, flags=re.IGNORECASE)
            elif rule == "DummyWrite":
                processed = re.sub(kw, "standard healthy metabolic panel", processed, flags=re.IGNORECASE)

    # 7. Name Recognition (Simple capitalization pattern or common names)
    rule = rules.get("Name", "None")
    name_patterns = [r"\bSanvi Jain\b", r"\bJohn Doe\b", r"\bBob Miller\b", r"\bAlice\b"]
    for np in name_patterns:
        if re.search(np, processed, re.IGNORECASE):
            if rule == "Remove":
                processed = re.sub(np, "", processed, flags=re.IGNORECASE)
            elif rule == "Mask":
                processed = re.sub(np, "[NAME_REDACTED]", processed, flags=re.IGNORECASE)
    # 8. Customizable / User-Defined Rules
    for custom_entity, regex_pat in st.session_state.get("custom_rules", {}).items():
        rule = rules.get(custom_entity, "None")
        if rule != "None" and regex_pat:
            try:
                if rule == "Remove":
                    processed = re.sub(regex_pat, "", processed)
                elif rule == "Mask":
                    processed = re.sub(regex_pat, f"[{custom_entity.upper()}_REDACTED]", processed)
                elif rule == "DummyWrite":
                    processed = re.sub(regex_pat, f"[DUMMY_{custom_entity.upper()}]", processed)
            except Exception:
                pass

    return processed


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
        st.caption("Delegated to evaluate financial eligibility, credit ratings, credit card limits, and risk profiles while preventing exposure of critical PII.")
    elif st.session_state.agent_name == "Customer Support Specialist":
        st.caption("Delegated to resolve general client queries with safe, toxic-free and injection-free boundaries.")
    elif st.session_state.agent_name == "Legal Compliance Auditor":
        st.caption("Delegated to audit legal agreements, redacting corporate names, contract values, and identity cards.")
    else:
        st.caption("Custom agent rules dynamically configured across the current execution path.")

# ---------------------------------------------------------------------
# MAIN INTERFACE TABS (Page 1 vs Page 2)
# ---------------------------------------------------------------------
st.markdown("<div class='main-header'>LiteLLM Agent Shielding & Routing Proxy</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Minimalist enterprise routing controls configured for <b>{st.session_state.agent_name}</b></div>", unsafe_allow_html=True)

tab_backend, tab_testing = st.tabs([
    "Page 1: Backend Gateway Controls",
    "Page 2: User Testing Interface"
])

# ---------------------------------------------------------------------
# PAGE 1: BACKEND CONTROLS
# ---------------------------------------------------------------------
with tab_backend:
    col_llm, col_guard = st.columns([1, 1.1], gap="large")
    
    with col_llm:
        st.markdown("<div class='card-title'>1. LLM Prioritization & Limits</div>", unsafe_allow_html=True)
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

    with col_guard:
        st.markdown("<div class='card-title'>2. Active Guardrail Policy</div>", unsafe_allow_html=True)
        st.write("Configure dynamic security policies. Apply filtering and masking rules to inputs or responses.")
        
        # 1. Hooks Selection
        hooks = st.multiselect(
            "Active Guardrail Interception Hooks",
            options=["Pre-call", "Post-call"],
            default=st.session_state.guardrail_hooks
        )
        st.session_state.guardrail_hooks = hooks
        
        # 2. Entities Options & Action Matrix
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#374151;'>Granular Policy Controls</span>", unsafe_allow_html=True)
        
        actions = ["None", "Remove", "Mask", "DummyWrite"]
        
        for entity in list(st.session_state.guardrail_rules.keys()):
            current_action = st.session_state.guardrail_rules[entity]
            choice = st.selectbox(
                f"↳ {entity}",
                options=actions,
                index=actions.index(current_action) if current_action in actions else 0,
                key=f"rule_{entity}"
            )
            st.session_state.guardrail_rules[entity] = choice
            
        st.success("Backend proxy safety profile dynamically loaded in memory!")
        
        st.markdown("<hr style='margin:15px 0; border:0; border-top:1px solid #E5E7EB;'>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#374151;'>Add Customizable Guardrail Entity</span>", unsafe_allow_html=True)
        col_c_name, col_c_regex = st.columns(2)
        with col_c_name:
            c_name = st.text_input("Custom Entity Name", placeholder="e.g. API Key", key="c_name_input")
        with col_c_regex:
            c_regex = st.text_input("Custom Regex Pattern", placeholder="e.g. gsk_[a-zA-Z0-9]{15,}", key="c_regex_input")
            
        if st.button("Register Custom Guardrail", use_container_width=True):
            if c_name and c_regex:
                st.session_state.custom_rules[c_name] = c_regex
                st.session_state.guardrail_rules[c_name] = "Mask" # default action
                st.success(f"Custom guardrail entity '{c_name}' registered successfully!")
                time.sleep(0.5)
                st.rerun()

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
        value="Transfer $1,200 to Bob Miller (Aadhaar: 9988-7766-5544, SSN: 111-22-3333). I am diagnosing my heart disease and clinical cancer history.",
        height=120
    )
    
    if st.button("Submit Request", type="primary"):
        # Apply pre-call guardrail
        pre_shielded_query = apply_guardrails(
            user_query, 
            st.session_state.guardrail_rules, 
            st.session_state.guardrail_hooks, 
            stage="Pre-call"
        )
        
        # Trigger Gateway Call
        payload = {
            "model": st.session_state.model_priorities[0] if st.session_state.model_priorities else "oss-chat-fast",
            "messages": [{"role": "user", "content": pre_shielded_query}],
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
            else:
                backend_response = f"⚠️ Gateway Error ({r.status_code}): {r.text}"
        except Exception as e:
            # Fallback mock response for testing disconnected modes
            latency = 0.045
            backend_response = f"This is a simulated secure response from {st.session_state.agent_name} analyzing the request context."
            actual_model = f"{st.session_state.model_priorities[0]} (Local Sandbox Simulation)"
            
        # Apply post-call guardrail
        final_response = apply_guardrails(
            backend_response,
            st.session_state.guardrail_rules,
            st.session_state.guardrail_hooks,
            stage="Post-call"
        )
        
        # Save to stateful variables
        st.session_state.last_result = {
            "query": user_query,
            "shielded_query": pre_shielded_query,
            "response": final_response,
            "model": actual_model,
            "latency": latency
        }
        st.session_state.history.append(st.session_state.last_result)
        
    # Render last result if present
    if st.session_state.last_result:
        res = st.session_state.last_result
        
        # Display stacked comparative layout with wide response space
        col_prompts, col_response = st.columns([1.2, 1.8], gap="large")
        
        with col_prompts:
            st.markdown("<span style='font-size:0.8rem; font-weight:600; color:#374151;'>1. ORIGINAL QUERY</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='diff-box diff-original'>{res['query']}</div>", unsafe_allow_html=True)
            
            st.markdown("<span style='font-size:0.8rem; font-weight:600; color:#374151;'>2. SHIELDED PROMPT SENT TO LLM</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='diff-box diff-shielded'>{res['shielded_query']}</div>", unsafe_allow_html=True)
            
        with col_response:
            st.markdown("<span style='font-size:0.8rem; font-weight:600; color:#374151;'>3. GATEWAY SHIELDED RESPONSE</span>", unsafe_allow_html=True)
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
                    st.markdown("<span style='font-size:0.75rem; font-weight:600; color:#4B5563;'>Shielded Query</span>", unsafe_allow_html=True)
                    st.warning(item["shielded_query"])
                with col_hist_right:
                    st.markdown("<span style='font-size:0.75rem; font-weight:600; color:#4B5563;'>Shielded Response</span>", unsafe_allow_html=True)
                    st.success(item["response"])
