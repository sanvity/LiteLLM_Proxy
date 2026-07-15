import streamlit as st
import requests
import os
import time
import re
import yaml
import json
import html
from synthetic_data import SyntheticDataEngine
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path, override=True)

def parse_tagged_text(tagged_text: str) -> dict:
    pattern = r'<([a-zA-Z0-9_\- ]+?)>(.*?)</\1>'
    clean_text = ""
    entities = []
    last_idx = 0
    
    for match in re.finditer(pattern, tagged_text):
        start_tagged, end_tagged = match.span()
        label = match.group(1)
        value = match.group(2)
        
        # Append part before match to clean text
        clean_text += tagged_text[last_idx:start_tagged]
        
        # Start and end of the entity in clean text
        entity_start = len(clean_text)
        clean_text += value
        entity_end = len(clean_text)
        
        entities.append({
            "start": entity_start,
            "end": entity_end,
            "label": label.strip().lower()
        })
        
        last_idx = end_tagged
        
    clean_text += tagged_text[last_idx:]
    return {"text": clean_text, "entities": entities}

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
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 500px;
        overflow-y: auto;
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

# Proxy URL configuration - supports dynamic environment overrides
PROXY_URL = (
    os.environ.get("PROXY_URL")
    or os.environ.get("FASTAPI_URL")
    or f"http://127.0.0.1:{os.environ.get('PORT', '8000')}"
).rstrip("/")

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

# Cached network operations to prevent blocking the Streamlit UI thread
@st.cache_data(ttl=10)
def check_backend_health(url):
    try:
        r = requests.get(f"{url}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=10)
def fetch_pii_config(url):
    try:
        r = requests.get(f"{url}/ui/pii-config", timeout=1.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"pii_enabled": False, "pii_action": "MASK", "pii_policy": {}}

@st.cache_data(ttl=5)
def fetch_training_status(url):
    try:
        r = requests.get(f"{url}/ui/train-deberta/status", timeout=1.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "idle", "progress": "", "error": None, "history": []}


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
if "training_completion_shown" not in st.session_state:
    st.session_state.training_completion_shown = True
if "view_report_id" not in st.session_state:
    st.session_state.view_report_id = None
    st.session_state.training_completion_shown = True

if "synthesis_inputs" not in st.session_state:
    st.session_state.synthesis_inputs = [
        {
            "id": 0,
            "data_format": "Medical Billing Invoice",
            "target_label": "patient id",
            "pattern_val": "PT-[0-9]{5}-[A-Z]{2}"
        }
    ]
    st.session_state.synthesis_inputs_counter = 1
else:
    # Migrate any existing entries without an ID and initialize the counter
    max_id = -1
    for idx, item in enumerate(st.session_state.synthesis_inputs):
        if "id" not in item:
            item["id"] = idx
        max_id = max(max_id, item["id"])
    if "synthesis_inputs_counter" not in st.session_state:
        st.session_state.synthesis_inputs_counter = max_id + 1
if "validation_report" not in st.session_state:
    st.session_state.validation_report = None
if "training_dataset" not in st.session_state:
    st.session_state.training_dataset = [
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

    st.markdown("---")
    st.markdown("<h4 style='font-weight:500; font-size:0.9rem; color:#4B5563;'>Backend Connection</h4>", unsafe_allow_html=True)
    st.code(PROXY_URL, language=None)
    
    is_connected = check_backend_health(PROXY_URL)
    if is_connected:
        st.success("✅ API Connected")
    else:
        st.warning("⚠️ API Unreachable")
        st.caption("The backend may be sleeping on Render. If so, it will take ~50s to wake up on the first request.")


# ---------------------------------------------------------------------
# MAIN INTERFACE TABS (Page 1 vs Page 2)
# ---------------------------------------------------------------------
st.markdown("<div class='main-header'>LiteLLM Gateway Console & Routing Proxy</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Minimalist enterprise routing controls configured for <b>{st.session_state.agent_name}</b></div>", unsafe_allow_html=True)

tab_backend, tab_testing, tab_training, tab_mlops = st.tabs([
    "Page 1: Backend Gateway Controls",
    "Page 2: User Testing Interface",
    "Page 3: Model Fine-Tuning",
    "Page 4: Model Registry & MLOps"
])

# ---------------------------------------------------------------------
# PAGE 1: BACKEND CONTROLS
# ---------------------------------------------------------------------
with tab_backend:
    col_llm, col_guardrails = st.columns([1, 1], gap="large")
    
    with col_llm:
        st.markdown("<div class='card-title'>LLM Prioritization & Limits</div>", unsafe_allow_html=True)
        st.write("Prioritize execution paths based on TPR/TPM loads. LiteLLM handles fallback routing dynamically.")
        
        # 1. LLM priorities via Drag-and-Drop sort_items
        st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#374151;'>Priority Routing Hierarchy (Drag & Drop to Order)</span>", unsafe_allow_html=True)
        st.caption("Drag models between containers to activate/deactivate, and reorder within the 'Active Priorities' list to set priority sequence.")
        
        from streamlit_sortables import sort_items
        
        active_list = st.session_state.model_priorities
        available_list = [m for m in configured_models if m not in active_list]
        
        sortable_data = [
            {'header': 'Active Priorities (Top is highest)', 'items': active_list},
            {'header': 'Available Models', 'items': available_list}
        ]
        
        sorted_data = sort_items(sortable_data, multi_containers=True, key="model_priority_sortable")
        
        if sorted_data is not None:
            new_priorities = sorted_data[0]['items']
            # Guarantee at least one active model if user dragged everything out
            if not new_priorities:
                new_priorities = [configured_models[0]]
            if new_priorities != st.session_state.model_priorities:
                st.session_state.model_priorities = new_priorities
                st.rerun()

            
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
        
        pii_cfg = fetch_pii_config(PROXY_URL)
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
            
        # Base known labels with user-friendly descriptions
        # Define the categories for standard/system labels
        STANDARD_PII_LABELS = {
            "person", "phone number", "social security number", "credit card number",
            "api key", "email address", "address", "bank account number", "passport number",
            "password", "age", "amount", "bic", "buildingnumber", "county", "currency",
            "currencycode", "currencysymbol", "date", "dob", "eyecolor", "gender",
            "height", "ip", "ipv4", "ipv6", "jobarea", "jobtitle", "jobtype", "mac",
            "maskednumber", "nearbygpscoordinate", "ordinaldirection", "pin", "prefix",
            "sex", "state", "time", "url", "useragent", "vehiclevin", "vehiclevrm"
        }
        
        BASIC_KEYS = ["person", "phone number", "email address", "social security number", "credit card number"]
        
        FRIENDLY_NAMES = {
            "person": "Name",
            "phone number": "Phone Number",
            "email address": "Email Address",
            "social security number": "SSN / Aadhaar",
            "credit card number": "Credit Card",
            "api key": "API Key",
            "address": "Address",
            "bank account number": "Bank Account Number",
            "passport number": "Passport Number",
            "password": "Password"
        }
        
        # Merge dynamically with active classes from fine-tuned model config
        active_labels = pii_cfg.get("active_labels", list(BASIC_KEYS))
        
        # Determine Top Keys: basics in active_labels + custom labels
        custom_keys = [k for k in active_labels if k not in STANDARD_PII_LABELS and k not in BASIC_KEYS]
        top_keys = [k for k in BASIC_KEYS if k in active_labels] + custom_keys
        
        CATEGORIES = {
            "Financial & Account Info": [
                "bank account number", "api key", "password", "amount", "currency", 
                "currencycode", "currencysymbol", "bic", "maskednumber", "pin"
            ],
            "Location, Address & Network": [
                "address", "buildingnumber", "county", "state", "nearbygpscoordinate", 
                "ip", "ipv4", "ipv6", "mac", "url"
            ],
            "Demographics & Personal Details": [
                "dob", "date", "time", "age", "gender", "sex", "eyecolor", "height"
            ],
            "Employment & Other Metadata": [
                "jobtitle", "jobarea", "jobtype", "employee_id", "patient_id", 
                "prefix", "useragent", "vehiclevin", "vehiclevrm", "passport number", "ordinaldirection"
            ]
        }
        
        pii_policy = {}
        
        # Helper function to render a dropdown selectbox for a specific label
        def render_policy_select(entity_key):
            entity_label = FRIENDLY_NAMES.get(entity_key, entity_key.replace("_", " ").title())
            if entity_key not in STANDARD_PII_LABELS and entity_key not in BASIC_KEYS:
                entity_label = f"{entity_label} [custom added]"
            default_action = pii_policy_default.get(entity_key, pii_action)
            options = ["BLOCK", "MASK", "REWRITE", "IGNORE"]
            selected_action = st.selectbox(
                entity_label,
                options=options,
                index=options.index(default_action) if default_action in options else options.index(pii_action),
                key=f"pii_action_{entity_key}"
            )
            pii_policy[entity_key] = selected_action
            
        # 1. Basics & Custom Labels at the top (directly visible)
        st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#374151; display:block; margin-bottom:8px;'>Core PII & Custom Labels</span>", unsafe_allow_html=True)
        col_ent1, col_ent2 = st.columns(2)
        for idx, entity_key in enumerate(top_keys):
            col = col_ent1 if idx % 2 == 0 else col_ent2
            with col:
                render_policy_select(entity_key)
                
        # 2. Categorized expanders for less common standard labels
        for cat_name, cat_keys in CATEGORIES.items():
            active_cat_keys = [k for k in cat_keys if k in active_labels and k not in top_keys]
            if active_cat_keys:
                with st.expander(f"📁 {cat_name}", expanded=False):
                    col_cat1, col_cat2 = st.columns(2)
                    for idx, entity_key in enumerate(active_cat_keys):
                        col = col_cat1 if idx % 2 == 0 else col_cat2
                        with col:
                            render_policy_select(entity_key)
                            
        # Handle any uncategorized standard labels just in case
        all_categorized = set(top_keys)
        for cat_keys in CATEGORIES.values():
            all_categorized.update(cat_keys)
        misc_keys = [k for k in active_labels if k not in all_categorized]
        if misc_keys:
            with st.expander("📁 Other PII Entities", expanded=False):
                col_misc1, col_misc2 = st.columns(2)
                for idx, entity_key in enumerate(misc_keys):
                    col = col_misc1 if idx % 2 == 0 else col_misc2
                    with col:
                        render_policy_select(entity_key)
                    
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
                    st.cache_data.clear()
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
        value="I am Samuel, my phone numbers are +1 213 555-0123 and +91 9876534567 , Aadhaar is 9988-7766-5544, SSN: 111-22-3333. my email is sam@gmail.com,  check which all accounts are linked together? "
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
            r = requests.post(f"{PROXY_URL}/v1/chat/completions", json=payload, timeout=60)
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
            backend_response = f"❌ Connection Error: Could not connect to the LiteLLM Proxy backend. Details: {e}"
            actual_model = "Unavailable"
            guardrailed_query = None
            latency = 0.0
            
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
            st.markdown(f"<div class='diff-box diff-original'>{html.escape(res['query'])}</div>", unsafe_allow_html=True)
            
            # Display guardrailed query below original query
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            st.markdown("<span style='font-size:0.8rem; font-weight:600; color:#374151;'>GUARDRAILED QUERY (SENT TO LLM)</span>", unsafe_allow_html=True)
            
            g_query = res.get("guardrailed_query")
            if g_query is None:
                st.markdown("<div class='diff-box diff-original' style='border-left-color: #EF4444; background-color: #FEF2F2; color: #EF4444; font-style: italic;'>[BLOCKED - NOT SENT TO LLM]</div>", unsafe_allow_html=True)
            elif g_query == res['query']:
                st.markdown(f"<div class='diff-box diff-shielded' style='border-left-color: #9CA3AF; background-color: #F9FAFB; color: #4B5563;'>{html.escape(g_query)}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='diff-box diff-shielded'>{html.escape(g_query)}</div>", unsafe_allow_html=True)
            
        with col_response:
            st.markdown("<span style='font-size:0.8rem; font-weight:600; color:#374151;'>GATEWAY RESPONSE</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='diff-box diff-response' style='min-height:220px;'>{html.escape(res['response'])}</div>", unsafe_allow_html=True)
            
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
    status_data = fetch_training_status(PROXY_URL)
        
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
        st.session_state.training_completion_shown = False
        st.info("⏳ Fine-tuning model in progress. Please wait...")
        history = status_data.get("history", [])
        if history:
            loss_steps = [log for log in history if "loss" in log]
            if loss_steps:
                st.write("📊 **Live Training Loss Curve**")
                st.line_chart([log["loss"] for log in loss_steps])
                
                # Show the latest stats
                latest_log = loss_steps[-1]
                col_live1, col_live2, col_live3 = st.columns(3)
                with col_live1:
                    st.metric("Current Step", latest_log.get("step", 0))
                with col_live2:
                    st.metric("Current Loss", f"{latest_log.get('loss', 0.0):.4f}")
                with col_live3:
                    st.metric("Epoch Progress", f"{latest_log.get('epoch', 0.0):.2f}")
        
        # Poll status in loop using progress indicator
        time.sleep(3.0)
        st.cache_data.clear() # clear cache to poll status
        st.rerun()
        
    elif current_status == "completed":
        st.success(
            "🎉 **Fine-tuning complete!** The DeBERTa PII model has been successfully updated with your custom training data. "
            "The active gateway proxy pipeline has automatically reloaded with the new weights.",
            icon="🎉"
        )
        if not st.session_state.get("training_completion_shown", False):
            st.toast("Model fine-tuning completed successfully!", icon="🎉")
            st.session_state.training_completion_shown = True
        
        # Pull history logs to visualize training effects
        history = status_data.get("history", [])
        loss_steps = [log for log in history if "loss" in log] if history else []
        
        col_m1, col_m2 = st.columns([1.1, 0.9], gap="large")
        
        with col_m1:
            st.markdown("<div class='card-title'>📈 Fine-Tuning Performance & Convergence</div>", unsafe_allow_html=True)
            if loss_steps:
                st.write("The chart below displays the loss reduction curve over the training steps:")
                st.line_chart([log["loss"] for log in loss_steps])
                
                # Stats summary
                final_loss = loss_steps[-1]["loss"]
                init_loss = loss_steps[0]["loss"]
                total_steps = len(loss_steps)
                reduction = ((init_loss - final_loss) / init_loss * 100) if init_loss > 0 else 0.0
                
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                col_st1, col_st2, col_st3 = st.columns(3)
                with col_st1:
                    st.metric("Total Steps", total_steps)
                with col_st2:
                    st.metric("Final Loss", f"{final_loss:.4f}", f"-{reduction:.1f}%")
                with col_st3:
                    # Try to fetch learning rate from last log
                    final_lr = loss_steps[-1].get("learning_rate", 1e-4)
                    st.metric("Final Learning Rate", f"{final_lr:.0e}")
            else:
                st.info("No training logs are available for the current fine-tuned weights run.")
                
        with col_m2:
            st.markdown("<div class='card-title'>🏷️ Active Model Entity Classifier Head</div>", unsafe_allow_html=True)
            st.write("All custom PII classes the model has been further trained to recognize locally:")
            
            # Fetch config for labels
            pii_cfg = fetch_pii_config(PROXY_URL)
            custom_labels = pii_cfg.get("custom_labels", [])
            
            if custom_labels:
                html_badges = []
                for lbl in custom_labels:
                    bg_style = "background-color: #F3E8FF; border: 1px solid #C084FC; color: #6B21A8;"
                    label_text = f"⭐ {lbl.upper()} (CUSTOM)"
                    html_badges.append(
                        f'<span style="{bg_style} padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.8rem; margin: 4px; display: inline-block;">'
                        f'{label_text}'
                        f'</span>'
                    )
                st.markdown(f'<div style="line-height: 2.2; margin-top: 10px;">{"".join(html_badges)}</div>', unsafe_allow_html=True)
            else:
                st.info("No custom entities trained yet. The model is currently using its pre-trained weights.")
                
        st.markdown("<hr style='margin:25px 0; border:0; border-top:1px solid #E5E7EB;'>", unsafe_allow_html=True)
        col_res1, _ = st.columns([1.5, 3.5])
        with col_res1:
            if st.button("🔄 Reset Console Status", type="primary", use_container_width=True):
                try:
                    requests.post(f"{PROXY_URL}/ui/train-deberta/reset", timeout=2.0)
                    st.cache_data.clear()
                    st.rerun()
                except Exception:
                    pass
                
    elif current_status == "failed":
        st.error("Training Traceback Log:")
        st.code(current_error or "Unknown error occurred.")
        if st.button("Clear Error & Reset"):
            try:
                requests.post(f"{PROXY_URL}/ui/train-deberta/reset", timeout=2.0)
                st.cache_data.clear()
                st.rerun()
            except Exception:
                pass
                
    else: # idle status - render configuration editor
        # 1. Synthesize Dataset Panel
        st.markdown("<div class='card-title'>Synthesize Balanced Training Dataset (Bias Prevention)</div>", unsafe_allow_html=True)
        st.write("Generate custom PII training data balanced dynamically to match the pre-trained model's multi-class distribution, avoiding single-label bias and catastrophic forgetting. Add multiple rows to synthesize varied data formats and labels in a single run.")
        
        inputs_list = st.session_state.synthesis_inputs
        
        for idx, item in enumerate(inputs_list):
            item_id = item["id"]
            st.markdown(f"<span style='font-size:0.85rem; font-weight:600; color:#4B5563; margin-top:10px; display:block;'>Custom PII Definition #{idx + 1}</span>", unsafe_allow_html=True)
            col_df, col_tl, col_pt, col_del = st.columns([3, 3, 3, 1])
            
            with col_df:
                df_val = st.text_input(
                    "Data Format / Domain Context",
                    value=item["data_format"],
                    key=f"df_{item_id}",
                    help="e.g. Medical Billing Invoice, Customer Support Ticket"
                )
            with col_tl:
                tl_val = st.text_input(
                    "Target PII Label to Train",
                    value=item["target_label"],
                    key=f"tl_{item_id}",
                    help="e.g. patient id, employee code"
                )
            with col_pt:
                pt_val = st.text_input(
                    "Alphanumeric Pattern / Seed (Optional)",
                    value=item["pattern_val"],
                    key=f"pt_{item_id}",
                    help="e.g. PT-[0-9]{5}-[A-Z]{2}"
                )
            with col_del:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                # Only allow deletion if there's more than one row
                if len(inputs_list) > 1:
                    if st.button("🗑️", key=f"del_{item_id}", help="Remove this definition"):
                        st.session_state.synthesis_inputs.pop(idx)
                        st.rerun()
            
            # Update values in-place directly in st.session_state
            item["data_format"] = df_val
            item["target_label"] = tl_val
            item["pattern_val"] = pt_val
        
        col_add, _ = st.columns([1.5, 3.5])
        with col_add:
            if st.button("➕ Add Another Definition", type="secondary", use_container_width=True):
                new_id = st.session_state.synthesis_inputs_counter
                st.session_state.synthesis_inputs.append({
                    "id": new_id,
                    "data_format": "Customer support log",
                    "target_label": "customer pin",
                    "pattern_val": "PIN-[0-9]{4}"
                })
                st.session_state.synthesis_inputs_counter += 1
                st.rerun()
                
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#374151;'>Dataset Generation Settings</span>", unsafe_allow_html=True)
        num_samples = st.number_input(
            "Number of Training Samples to Generate",
            min_value=10,
            max_value=5000,
            value=50,
            step=10,
            help="No more timeouts! The new Diversity-Driven engine runs iteratively one-by-one, enabling generation of large datasets (up to 5000+ samples) completely safely."
        )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            
        if st.button("Synthesize Dataset", type="secondary", use_container_width=True):
            # Validate all fields
            valid_inputs = True
            for idx, item in enumerate(st.session_state.synthesis_inputs):
                if not item["data_format"].strip() or not item["target_label"].strip():
                    st.error(f"Please provide both a Data Format and a Target PII Label for Definition #{idx + 1}.")
                    valid_inputs = False
                    break
                    
            if valid_inputs:
                # Retrieve target labels
                allowed_labels = ["person", "email address", "phone number", "address", "credit card number", "social security number", "passport number", "bank account number", "password", "api key"]
                for item in st.session_state.synthesis_inputs:
                    lbl = item["target_label"].strip().lower()
                    if lbl not in allowed_labels:
                        allowed_labels.append(lbl)
                
                # Import and execute our Diversity-Driven Synthetic Data Engine
                try:
                    from synthetic_data import SyntheticDataEngine
                    engine = SyntheticDataEngine()
                    
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    def update_progress(current, total, stats):
                        pct = current / total
                        progress_bar.progress(pct)
                        healed = stats.get("healed_entities", 0)
                        attempts = stats.get("total_attempts", 0)
                        status_text.markdown(
                            f"🤖 **Generating sample {current}/{total}...** (Attempts: {attempts} | Auto-healed Spans: {healed})"
                        )
                    
                    model_to_use = st.session_state.model_priorities[0] if st.session_state.model_priorities else "groq/llama-3.1-8b-instant"
                    
                    # Run generation
                    with st.spinner("Executing Diversity-Driven Synthetic Data Engine..."):
                        dataset = engine.generate_dataset(
                            num_samples=int(num_samples),
                            target_labels=allowed_labels,
                            model=model_to_use,
                            progress_callback=update_progress,
                            synthesis_inputs=st.session_state.synthesis_inputs
                        )
                        
                    if len(dataset) > 0:
                        st.session_state.training_dataset = dataset
                        st.session_state.validation_report = getattr(dataset, "report", None)
                        st.success(f"Synthesized {len(dataset)} balanced, diversity-driven training samples successfully!")
                        st.balloons()
                        time.sleep(1.0)
                        st.rerun()
                    else:
                        st.error("No samples were successfully generated. Please check model and API configurations.")
                except Exception as ex:
                    st.error(f"Generation Engine failed: {ex}")

        st.markdown("<hr style='margin:20px 0; border:0; border-top:1px solid #E5E7EB;'>", unsafe_allow_html=True)

        # Training Hyperparameters
        st.markdown("<div class='card-title'>Training Hyperparameters</div>", unsafe_allow_html=True)
        st.write("Configure the fine-tuning run parameters before submitting the training job.")

        col_hp1, col_hp2, col_hp3 = st.columns(3)
        with col_hp1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=100,
                value=15,
                step=1,
                help="Number of full passes through the training dataset. More epochs = longer training but better fit. Recommended: 10–20."
            )
        with col_hp2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                value=1e-4,
                format_func=lambda x: f"{x:.0e}",
                help="Step size for gradient updates. Lower = slower but more stable. Recommended: 1e-4 for fine-tuning DeBERTa."
            )
        with col_hp3:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=64,
                value=8,
                step=1,
                help="Number of samples processed per gradient step. Larger = faster but uses more memory. Recommended: 8."
            )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.caption(f"📅 **{epochs}** training epoch{'s' if epochs != 1 else ''}")
        with col_info2:
            st.caption(f"📉 Learning rate: **{learning_rate:.0e}**")
        with col_info3:
            st.caption(f"📦 Batch size: **{batch_size}** samples/step")

        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#374151;'>Hyperparameter Tuning (Optuna)</span>", unsafe_allow_html=True)
        col_optuna_enable, col_optuna_trials = st.columns([1, 2])
        with col_optuna_enable:
            use_optuna = st.checkbox(
                "Enable Optuna Tuning",
                value=False,
                help="Search for the best learning rate, batch size, and epochs automatically before training. This can improve model performance."
            )
        with col_optuna_trials:
            optuna_trials = st.slider(
                "Number of Search Trials",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                disabled=not use_optuna,
                help="Recommended: 3. Higher numbers search more combinations but take longer."
            )

        if use_optuna:
            st.info(f"🔍 Optuna tuning is enabled. Epochs, Learning Rate, and Batch Size inputs above will be overridden by the best parameters discovered during the {optuna_trials} search trials.")

        # 3. Input Training Dataset
        st.markdown("<div class='card-title'>Input Training Dataset (JSON Format)</div>", unsafe_allow_html=True)
        st.write("Define text training samples and label offsets for NER Token Classification. Matches canonical labels.")
        
        if st.session_state.get("validation_report"):
            rep = st.session_state.validation_report
            st.markdown(f"""
            <div class='card' style='background-color:#F0FDF4; padding:12px; border-radius:6px; margin-bottom:12px; border:1px solid #BBF7D0;'>
                <span style='font-size:0.9rem; font-weight:600; color:#166534;'>📊 Generation Validation Report</span>
                <div style='display:flex; justify-content:space-between; margin-top:8px; font-size:0.85rem; color:#14532D;'>
                    <div><b>Valid Samples</b>: {rep.get('sample_counts', 0)}</div>
                    <div><b>Dropped</b>: {rep.get('dropped_count', 0)}</div>
                    <div><b>Hard Negatives</b>: {rep.get('hard_negatives_count', 0)} ({rep.get('percent_hard_negatives', 0.0):.1f}%)</div>
                    <div><b>Near-Dup Rate</b>: {rep.get('near_dup_rate', 0.0)*100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        dataset_json = st.text_area(
            "Training Dataset JSON",
            value=json.dumps(st.session_state.training_dataset, indent=2),
            height=300
        )

        # 4. Bias Prevention Checklist & Analytics
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        try:
            active_dataset = json.loads(dataset_json)
            if isinstance(active_dataset, list):
                total_samples = len(active_dataset)
                if total_samples > 0:
                    label_occurrences = {}
                    neutral_count = 0
                    
                    for sample in active_dataset:
                        ents = sample.get("entities", [])
                        if not ents:
                            neutral_count += 1
                        for ent in ents:
                            lbl = ent.get("label", "unknown").lower()
                            label_occurrences[lbl] = label_occurrences.get(lbl, 0) + 1
                    
                    st.markdown("<div class='card' style='background-color:#F9FAFB; padding:12px; border-radius:6px;'>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#374151;'>Active Dataset Balance & Bias Check</span>", unsafe_allow_html=True)
                    col_b1, col_b2 = st.columns([1, 2])
                    with col_b1:
                        st.write(f"**Total Samples**: {total_samples}")
                        st.write(f"**Neutral Samples (no PII)**: {neutral_count} ({neutral_count/total_samples*100:.1f}%)")
                    with col_b2:
                        if label_occurrences:
                            breakdown_items = [f"**{lbl}**: {count} ({count/total_samples*100:.1f}% of samples)" for lbl, count in label_occurrences.items()]
                            st.write("**Entity Instances Frequency**:")
                            for item in breakdown_items:
                                st.write(f"• {item}")
                        else:
                            st.write("• No PII entities annotated yet.")
                    st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            pass
        
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
                            "batch_size": int(batch_size),
                            "use_optuna": bool(use_optuna),
                            "optuna_trials": int(optuna_trials)
                        }
                        r = requests.post(f"{PROXY_URL}/ui/train-deberta", json=payload, timeout=5.0)
                        if r.status_code == 200:
                            st.success("Training task submitted successfully. Starting background threads...")
                            st.session_state.training_completion_shown = False
                            st.cache_data.clear()
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Failed to submit training job: {r.text}")
            except json.JSONDecodeError as je:
                st.error(f"Invalid JSON Syntax: {je}")
            except Exception as e:
                st.error(f"Submission failed: {e}")


# ---------------------------------------------------------------------
# PAGE 4: MODEL REGISTRY & MLOPS
# ---------------------------------------------------------------------
with tab_mlops:
    st.markdown("<div class='card-title'>MLOps Model Version Control Registry</div>", unsafe_allow_html=True)
    st.write("Track, evaluate, and deploy fine-tuned DeBERTa model versions. Switch between versions dynamically.")
    
    # 1. Fetch registry state from backend
    registry_data = {"active_version": None, "versions": []}
    try:
        r = requests.get(f"{PROXY_URL}/ui/mlops/registry", timeout=2.0)
        if r.status_code == 200:
            registry_data = r.json()
    except Exception as e:
        st.error(f"Failed to connect to model registry: {e}")
        
    active_version = registry_data.get("active_version")
    versions = registry_data.get("versions", [])
    
    # Let's check background evaluation status
    eval_status = {"status": "idle", "progress": "", "error": None}
    try:
        r_eval = requests.get(f"{PROXY_URL}/ui/mlops/evaluate/status", timeout=1.0)
        if r_eval.status_code == 200:
            eval_status = r_eval.json()
    except Exception:
        pass
        
    if eval_status["status"] == "running":
        st.info(f"⏳ **Model Evaluation in Progress**: {eval_status['progress']}")
        time.sleep(2.0)
        st.cache_data.clear()
        st.rerun()
    elif eval_status["status"] == "completed":
        st.success("🎉 Evaluation completed successfully!")
        if st.button("Clear Evaluation Notification"):
            st.rerun()
    elif eval_status["status"] == "failed":
        st.error(f"❌ Evaluation failed: {eval_status['error']}")
        
    # Active Model HUD Card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col_hud1, col_hud2, col_hud3 = st.columns(3)
    with col_hud1:
        st.markdown("<span class='metadata-label'>Active Model Deploy Status</span>", unsafe_allow_html=True)
        if active_version:
            st.markdown(f"<h3 style='color:#10B981; margin-top:5px; font-weight:600;'>{active_version}</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:#4B5563; margin-top:5px; font-weight:600;'>Base Model / Staging</h3>", unsafe_allow_html=True)
    with col_hud2:
        st.markdown("<span class='metadata-label'>Registry Saved Count</span>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#3B82F6; margin-top:5px; font-weight:600;'>{len(versions)}</h3>", unsafe_allow_html=True)
    with col_hud3:
        st.markdown("<span class='metadata-label'>PII Guardrail Status</span>", unsafe_allow_html=True)
        is_pii_enabled = fetch_pii_config(PROXY_URL).get("pii_enabled", False)
        if is_pii_enabled:
            st.markdown("<h3 style='color:#10B981; margin-top:5px; font-weight:600;'>🛡️ ACTIVE</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:#EF4444; margin-top:5px; font-weight:600;'>⚠️ DISABLED</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Staging Model Section
    st.markdown("<h4 style='font-weight:600; color:#111827;'>Staging Model (Staged Training Weights)</h4>", unsafe_allow_html=True)
    st.caption("This is the directory containing the latest fine-tuning weights before saving them to the registry.")
    
    # Verify if staging model has config.json
    staging_exists = False
    try:
        r_stage = requests.get(f"{PROXY_URL}/ui/mlops/report/staging", timeout=1.0)
        staging_exists = (r_stage.status_code != 404)
    except Exception:
        pass
        
    if staging_exists or current_status == "completed":
        st.markdown("<div class='card' style='background-color: #F8FAFC;'>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:0.9rem; font-weight:600; color:#4B5563;'>Staged Model in models/finetuned-deberta</span>", unsafe_allow_html=True)
        
        col_st_act1, col_st_act2, col_st_act3 = st.columns(3)
        with col_st_act1:
            if st.button("📊 Evaluate Staging", use_container_width=True):
                r = requests.post(f"{PROXY_URL}/ui/mlops/evaluate", json={"version_id": "staging"}, timeout=3.0)
                if r.status_code == 200:
                    st.success("Staging model evaluation started.")
                    st.rerun()
                else:
                    st.error(f"Failed to evaluate: {r.text}")
        with col_st_act2:
            if st.button("🚀 Deploy Staging (Staging Mode)", use_container_width=True):
                r = requests.post(f"{PROXY_URL}/ui/mlops/deploy", json={"version_id": None}, timeout=3.0)
                if r.status_code == 200:
                    st.success("Deployed staging model.")
                    st.rerun()
                else:
                    st.error(f"Failed to deploy: {r.text}")
        with col_st_act3:
            # Check if report exists
            has_report = False
            try:
                r_rep = requests.get(f"{PROXY_URL}/ui/mlops/report/staging", timeout=1.0)
                has_report = (r_rep.status_code == 200)
            except Exception:
                pass
            if has_report:
                if st.button("📄 View Staging Report", use_container_width=True):
                    st.session_state.view_report_id = "staging"
            else:
                st.button("📄 View Staging Report", disabled=True, use_container_width=True)
                
        # Form to save staging to registry
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.form("save_to_registry_form"):
            st.markdown("<span style='font-size: 0.8rem; font-weight: 600; color: #4B5563;'>Save Staging model to Registry</span>", unsafe_allow_html=True)
            v_name = st.text_input("Version Tag Name", placeholder="e.g. Finance V1")
            v_desc = st.text_area("Version Description", placeholder="Trained with 100 invoice samples to detect client accounts.")
            
            # Extract dataset size from app.py state if available
            dataset_size = len(st.session_state.training_dataset)
            
            submit_save = st.form_submit_button("💾 Save Model to Registry", use_container_width=True)
            if submit_save:
                if not v_name.strip():
                    st.error("Version Tag Name is required.")
                else:
                    payload = {
                        "name": v_name,
                        "description": v_desc,
                        "config": {
                            "epochs": int(epochs) if "epochs" in locals() else 15,
                            "learning_rate": float(learning_rate) if "learning_rate" in locals() else 1e-4,
                            "batch_size": int(batch_size) if "batch_size" in locals() else 8,
                            "dataset_size": dataset_size
                        }
                    }
                    r = requests.post(f"{PROXY_URL}/ui/mlops/save", json=payload, timeout=5.0)
                    if r.status_code == 200:
                        st.success("Model version saved successfully to registry!")
                        st.rerun()
                    else:
                        st.error(f"Failed to save version: {r.text}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No staging model weights exist. Go to Tab 3: Model Fine-Tuning to run a training loop first.")

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-weight:600; color:#111827;'>Saved Model Registry & Version Control</h4>", unsafe_allow_html=True)
    
    if len(versions) == 0:
        st.info("No versions saved in registry yet.")
    else:
        for idx, ver in enumerate(versions):
            vid = ver["id"]
            vname = ver["name"]
            vdesc = ver["description"]
            created = ver["created_at"]
            status = ver["status"]
            vmetrics = ver.get("metrics")
            
            # Format display card
            is_active = (status == "active")
            card_border = "2px solid #10B981" if is_active else "1px solid #E5E7EB"
            bg_color = "#F0FDF4" if is_active else "#FFFFFF"
            
            st.markdown(f"""
            <div style="border: {card_border}; border-radius: 8px; padding: 1.25rem; background-color: {bg_color}; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 600; color: #111827;">{vname} <span style="font-size: 0.8rem; color:#6B7280; font-weight:400;">({vid})</span></span>
                    {"<span style='background-color: #D1FAE5; color: #065F46; padding: 3px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.75rem;'>ACTIVE/DEPLOYED</span>" if is_active else "<span style='background-color: #F3F4F6; color: #374151; padding: 3px 10px; border-radius: 9999px; font-weight: 500; font-size: 0.75rem;'>INACTIVE</span>"}
                </div>
                <div style="color: #4B5563; font-size: 0.9rem; margin-bottom: 12px;">{vdesc}</div>
                <div style="display: flex; gap: 20px; font-size: 0.8rem; color: #6B7280; margin-bottom: 15px;">
                    <div>📅 Created: <b>{created}</b></div>
                    <div>📦 Configs: Epochs=<b>{ver.get('epochs')}</b>, LR=<b>{ver.get('learning_rate')}</b>, Batch=<b>{ver.get('batch_size')}</b></div>
                    <div>📊 Dataset size: <b>{ver.get('dataset_size')} samples</b></div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display metrics if evaluated
            if vmetrics and isinstance(vmetrics, dict):
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric("Macro F1", f"{vmetrics.get('f1', 0.0):.3f}")
                with col_met2:
                    st.metric("Precision", f"{vmetrics.get('precision', 0.0):.3f}")
                with col_met3:
                    st.metric("Recall", f"{vmetrics.get('recall', 0.0):.3f}")
            else:
                st.warning("⚠️ This version has not been evaluated on the held-out test set yet.")
                
            # Actions buttons for this version
            col_vact1, col_vact2, col_vact3, col_vact4 = st.columns(4)
            with col_vact1:
                if is_active:
                    st.button("🚀 Deployed", disabled=True, key=f"dep_btn_dis_{vid}", use_container_width=True)
                else:
                    if st.button("🚀 Deploy", key=f"dep_btn_{vid}", use_container_width=True):
                        r = requests.post(f"{PROXY_URL}/ui/mlops/deploy", json={"version_id": vid}, timeout=3.0)
                        if r.status_code == 200:
                            st.success(f"Version {vname} deployed!")
                            st.rerun()
                        else:
                            st.error(f"Failed to deploy: {r.text}")
            with col_vact2:
                if st.button("📊 Evaluate", key=f"eval_btn_{vid}", use_container_width=True):
                    r = requests.post(f"{PROXY_URL}/ui/mlops/evaluate", json={"version_id": vid}, timeout=3.0)
                    if r.status_code == 200:
                        st.success(f"Evaluation started for {vname}.")
                        st.rerun()
                    else:
                        st.error(f"Failed to evaluate: {r.text}")
            with col_vact3:
                # Check if report exists
                has_report = False
                try:
                    r_rep = requests.get(f"{PROXY_URL}/ui/mlops/report/{vid}", timeout=1.0)
                    has_report = (r_rep.status_code == 200)
                except Exception:
                    pass
                if has_report:
                    if st.button("📄 View Report", key=f"view_rep_{vid}", use_container_width=True):
                        st.session_state.view_report_id = vid
                        st.rerun()
                else:
                    st.button("📄 View Report", key=f"view_rep_dis_{vid}", disabled=True, use_container_width=True)
            with col_vact4:
                if st.button("🗑️ Delete", key=f"del_btn_{vid}", use_container_width=True):
                    r = requests.post(f"{PROXY_URL}/ui/mlops/delete", json={"version_id": vid}, timeout=3.0)
                    if r.status_code == 200:
                        st.success(f"Version {vname} deleted.")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete: {r.text}")
                        
            st.markdown("</div>", unsafe_allow_html=True)
            
    # Iframe rendering if a report has been selected for viewing
    if st.session_state.get("view_report_id"):
        view_id = st.session_state.view_report_id
        st.markdown("<hr style='margin:30px 0; border:0; border-top:1px solid #E5E7EB;'>", unsafe_allow_html=True)
        st.markdown(f"### 📄 Evaluation Report Viewer: {view_id}")
        if st.button("❌ Close Report Viewer"):
            st.session_state.view_report_id = None
            st.rerun()
            
        report_url = f"{PROXY_URL}/ui/mlops/report/{view_id}"
        st.components.v1.iframe(report_url, height=800, scrolling=True)

