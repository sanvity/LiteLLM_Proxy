import os
import sys
import time
import requests
import asyncio
import unittest
import threading
import logging
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

# Set up logging for testing
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("proxy.test")

# Ensure .env is loaded
load_dotenv()

from proxy import ProxyConfig, LiteLLMProxyRouter, LiteLLMProxyApp

class TestLiteLLMProxy(unittest.TestCase):
    """
    Complete unit and integration test suite for the LiteLLM Proxy.
    Validates PII shielding, config parsing, token rates, and fallback multi-cluster routing.
    """
    
    @classmethod
    def setUpClass(cls):
        logger.info("Initializing OOP LiteLLM Proxy components for verification...")
        
        # Clean up any existing local model directory to ensure clean start
        import shutil
        current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        local_model_dir = os.path.join(current_dir, "models", "finetuned-deberta")
        if os.path.exists(local_model_dir):
            logger.info("Deleting existing local model directory for clean test start...")
            try:
                shutil.rmtree(local_model_dir)
            except Exception as e:
                logger.warning(f"Could not delete local model directory: {e}")
                
        # Clean up any existing pii_guardrail_config.json to ensure fresh default behavior
        pii_config_path = os.path.join(current_dir, "config", "pii_guardrail_config.json")
        if os.path.exists(pii_config_path):
            logger.info("Deleting existing pii_guardrail_config.json for clean test start...")
            try:
                os.remove(pii_config_path)
            except Exception as e:
                logger.warning(f"Could not delete pii_guardrail_config.json: {e}")
        
        cls.config = ProxyConfig(config_path="config.yaml")
        cls.router = LiteLLMProxyRouter(config=cls.config)

    def test_1_config_loading(self):
        """Verifies that config.yaml endpoints and global router parameters are parsed correctly into Pydantic models."""
        logger.info("--- Test 1: Configuration Loading ---")
        self.assertGreater(len(self.config.endpoints), 0, "No model endpoints loaded!")
        
        # Check router configurations (aligned to user settings)
        self.assertEqual(self.config.routing_strategy, "usage-based-routing")
        self.assertEqual(self.config.num_retries, 3)
        self.assertEqual(self.config.timeout, 10)
        
        # Inspect loaded endpoints
        primary_nodes = self.config.get_endpoints_for_model("primary-cluster")
        backup_nodes = self.config.get_endpoints_for_model("backup-cluster")
        local_nodes = self.config.get_endpoints_for_model("local-fallback-cluster")
        
        self.assertGreater(len(primary_nodes), 0, "No 'primary-cluster' endpoints found.")
        self.assertGreater(len(backup_nodes), 0, "No 'backup-cluster' endpoints found.")
        
        logger.info(f"Loaded: {len(primary_nodes)} primary nodes, {len(backup_nodes)} backups, and {len(local_nodes)} local fallbacks.")

    def test_2_token_estimation(self):
        """Tests that tiktoken and the character-count fallbacks work properly."""
        logger.info("--- Test 2: Token Estimation ---")
        text = "This is a simple prompt for verifying the token estimator."
        
        # Test basic estimation
        token_count = self.router.estimate_tokens(text)
        self.assertGreater(token_count, 0)
        
        # Test message array estimation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a story about gravity."}
        ]
        total_tokens = self.router.estimate_request_tokens(messages)
        self.assertGreater(total_tokens, len(messages) * 5)
        logger.info(f"Prompt content: {len(text)} chars -> estimated {token_count} tokens.")
        logger.info(f"System+User chat list -> estimated {total_tokens} tokens.")

    def test_3_tpr_context_filtering(self):
        """
        Tests Tokens Per Request (TPR) context window filtering and escalation in mock sandbox mode.
        A short prompt should route to primary Llama-3.1-8B (context limit 8K for Cerebras).
        An extremely long prompt that exceeds 8K context should automatically route
        to the high-capacity backup-cluster (128K context window).
        """
        logger.info("--- Test 3: Tokens Per Request (TPR) Dynamic Selection ---")
        
        # Case A: Short Prompt (Fits in Cerebras 8K context)
        messages_short = [{"role": "user", "content": "Hello, how are you today?"}]
        response_short = asyncio.run(self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages_short,
            max_tokens=100,
            mock_sandbox=True
        ))
        model_short = response_short["model"]
        self.assertIn(model_short, [
            "groq/llama-3.1-8b-instant", 
            "cerebras/llama3.1-8b"
        ])
        logger.info(f"Short prompt routed correctly to primary node: {model_short}")
        
        # Case B: Huge Prompt (Exceeds Llama 3.1 8B's 131k context limit)
        messages_large = [{"role": "user", "content": "Explain gravity in great detail." * 1000}]
        response_large = asyncio.run(self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages_large,
            max_tokens=130000,
            mock_sandbox=True
        ))
        model_large = response_large["model"]
        self.assertEqual(model_large, "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo")
        logger.info(f"Huge prompt auto-escalated correctly in sandbox to backup premium cluster: {model_large}")
 
    def test_4_mock_sandbox_completion(self):
        """Verifies that we can trigger the router completion flow in mock sandbox mode to test load balancing splits."""
        logger.info("--- Test 4: Mock Sandbox Load-Balancing Execution ---")
        
        messages = [{"role": "user", "content": "Compute the trajectory of a falling apple."}]
        
        response = asyncio.run(self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages,
            max_tokens=100,
            mock_sandbox=True
        ))
        
        self.assertIn("id", response)
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(len(response["choices"]), 1)
        self.assertEqual(response["choices"][0]["message"]["role"], "assistant")
        
        content = response["choices"][0]["message"]["content"]
        self.assertIn("[LiteLLM Proxy Mock -", content)
        
        # Inspect metrics tracking
        metrics = self.router.get_metrics()
        self.assertGreater(metrics["successful_requests"], 0)
        self.assertGreater(metrics["total_input_tokens"], 0)
        
        logger.info(f"Completion Mock successful. Response returned:\n{content}")
        logger.info(f"Updated proxy metrics: {metrics}")

    def test_5_api_microservice_integration(self):
        """
        Integration test: Starts the FastAPI microservice in a background thread,
        and fires HTTP client requests against standard OpenAI paths.
        """
        logger.info("--- Test 5: FastAPI Microservice Integration Test ---")
        
        # Initialize the microservice container on port 8090 to avoid collisions
        os.environ["PORT"] = "8090"
        proxy_app = LiteLLMProxyApp(config_path="config.yaml")
        app_instance = proxy_app.get_app()
        
        import uvicorn
        
        # Run uvicorn in a background thread
        server_thread = threading.Thread(
            target=lambda: uvicorn.run(app_instance, host="127.0.0.1", port=8090, log_level="warning"),
            daemon=True
        )
        server_thread.start()
        
        # Wait a moment for server boot
        time.sleep(1.5)
        
        base_url = "http://127.0.0.1:8090"
        
        # 1. Verify health check
        health_resp = requests.get(f"{base_url}/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertEqual(health_resp.json()["status"], "healthy")
        logger.info(f"Health check verified: {health_resp.json()}")
        
        # 2. Verify models list
        models_resp = requests.get(f"{base_url}/v1/models")
        self.assertEqual(models_resp.status_code, 200)
        models_list = models_resp.json()["data"]
        self.assertGreater(len(models_list), 0)
        logger.info(f"Models list endpoint verified. Found {len(models_list)} registered models.")
        
        # 3. Verify chat completion request via HTTP client using mock_sandbox
        payload = {
            "model": "primary-cluster",
            "messages": [
                {"role": "user", "content": "What is the escape velocity of Earth?"}
            ],
            "max_tokens": 150,
            "mock_sandbox": True
        }
        
        completion_resp = requests.post(f"{base_url}/v1/chat/completions", json=payload)
        self.assertEqual(completion_resp.status_code, 200)
        completion_data = completion_resp.json()
        
        self.assertIn("choices", completion_data)
        reply = completion_data["choices"][0]["message"]["content"]
        self.assertIn("LiteLLM Proxy Mock", reply)
        logger.info(f"Chat completion HTTP endpoint verified. Response:\n{reply}")
        
        # 4. Verify metrics endpoint tracks real-time usage
        metrics_resp = requests.get(f"{base_url}/metrics")
        self.assertEqual(metrics_resp.status_code, 200)
        metrics_data = metrics_resp.json()
        self.assertGreater(metrics_data["metrics"]["successful_requests"], 0)
        logger.info(f"Metrics endpoint verified. Captured stats: {metrics_data['metrics']}")

    def test_6_complexity_routing(self):
        """
        Tests that prompt complexity is correctly classified and routes requests
        to the appropriate complexity-tiered models.
        """
        logger.info("--- Test 6: Complexity-Aware Routing ---")
        
        # Case A: Low Complexity prompt (Simple greeting)
        # Should match 'low' complexity and route to a low cost node (e.g. cerebras/llama3.1-8b or ollama/llama3.1)
        messages_low = [{"role": "user", "content": "Hello! How is it going?"}]
        response_low = asyncio.run(self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages_low,
            max_tokens=100,
            mock_sandbox=True
        ))
        model_low = response_low["model"]
        # With low complexity prompt, the best node under primary-cluster is Cerebras Llama 3.1 8B (tier: low, cost: 0.01)
        self.assertEqual(model_low, "cerebras/llama3.1-8b")
        logger.info(f"Low complexity prompt routed correctly to low-cost node: {model_low}")
        
        # Case B: High Complexity prompt (Coding / optimization request)
        # Should match 'high' complexity and route to a high reasoning node (e.g. groq/llama-3.3-70b-versatile)
        messages_high = [{"role": "user", "content": "Optimize this SQL query and write a python refactoring function to execute it cleanly."}]
        response_high = asyncio.run(self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages_high,
            max_tokens=150,
            mock_sandbox=True
        ))
        model_high = response_high["model"]
        # With high complexity prompt, the best node under primary-cluster is Groq Llama 3.3 70B (tier: high, cost: 0.70)
        self.assertEqual(model_high, "groq/llama-3.3-70b-versatile")
        logger.info(f"High complexity prompt routed correctly to reasoning node: {model_high}")
        
        # Case C: Medium Complexity prompt (Summarization task)
        # Should match 'medium' complexity and route to a medium tier node (e.g. groq/llama-3.1-8b-instant)
        messages_med = [{"role": "user", "content": "Please write a concise summarization of this company's Q1 financial report."}]
        response_med = asyncio.run(self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages_med,
            max_tokens=150,
            mock_sandbox=True
        ))
        model_med = response_med["model"]
        # With medium complexity prompt, the best node under primary-cluster is Groq Llama 3.1 8B (tier: medium, cost: 0.05)
        self.assertEqual(model_med, "groq/llama-3.1-8b-instant")
        logger.info(f"Medium complexity prompt routed correctly to standard node: {model_med}")

    def test_7_priority_preference_routing(self):
        """
        Tests the Priority-Based Preference Routing and Credit Limit Failover:
        - Configures custom preference order: [groq/llama-3.1-8b-instant, cerebras/llama3.1-8b]
        - Sets tight budget limits.
        - Verifies it routes requests to the first preferred model.
        - Exceeds the first model's budget, then verifies it automatically cascades/fails over to the second!
        - Resets the spend, verifying it goes back to the first.
        """
        logger.info("--- Test 7: Priority-Based Preference Routing & Credit Limits ---")
        
        # Save original preference state
        old_pref_enabled = self.router.preference_enabled
        old_pref_list = self.router.preference_list
        old_limits = self.router.credit_limits.copy()
        old_spend = self.router.accumulated_spend.copy()
        
        try:
            self.router.preference_enabled = True
            self.router.preference_list = [
                "groq/llama-3.1-8b-instant",
                "cerebras/llama3.1-8b"
            ]
            
            # Reset spend counters
            self.router.accumulated_spend["groq/llama-3.1-8b-instant"] = 0.0
            self.router.accumulated_spend["cerebras/llama3.1-8b"] = 0.0
            
            # Set tiny credit limit for first priority model
            self.router.credit_limits["groq/llama-3.1-8b-instant"] = 0.000001
            self.router.credit_limits["cerebras/llama3.1-8b"] = 0.05
            
            # 1. Fire first request - should route to Priority 1 (groq/llama-3.1-8b-instant)
            messages = [{"role": "user", "content": "Test prompt sequence"}]
            response_1 = asyncio.run(self.router.execute_chat_completion(
                model="primary-cluster",
                messages=messages,
                max_tokens=100,
                mock_sandbox=True
            ))
            
            self.assertEqual(response_1["model"], "groq/llama-3.1-8b-instant")
            logger.info("Successfully routed first request to Priority 1 model.")
            
            # 2. Check spend has increased and exceeded the $0.000001 limit
            groq_spend = self.router.accumulated_spend.get("groq/llama-3.1-8b-instant", 0.0)
            self.assertGreater(groq_spend, 0.000001)
            logger.info(f"Priority 1 spend accumulated correctly: ${groq_spend:.6f} > $0.00001 limit.")
            
            # 3. Fire second request - should automatically fail over / cascade to Priority 2 (cerebras/llama3.1-8b)
            response_2 = asyncio.run(self.router.execute_chat_completion(
                model="primary-cluster",
                messages=messages,
                max_tokens=100,
                mock_sandbox=True
            ))
            self.assertEqual(response_2["model"], "cerebras/llama3.1-8b")
            logger.info("Successfully failed over to Priority 2 model once Priority 1 budget was exceeded.")
            
            # 4. Reset spend and verify it routes back to Priority 1 (groq/llama-3.1-8b-instant)
            self.router.accumulated_spend["groq/llama-3.1-8b-instant"] = 0.0
            response_3 = asyncio.run(self.router.execute_chat_completion(
                model="primary-cluster",
                messages=messages,
                max_tokens=100,
                mock_sandbox=True
            ))
            self.assertEqual(response_3["model"], "groq/llama-3.1-8b-instant")
            logger.info("Successfully routed back to Priority 1 after resetting budget spend counters.")
            
        finally:
            # Restore original state
            self.router.preference_enabled = old_pref_enabled
            self.router.preference_list = old_pref_list
            self.router.credit_limits = old_limits
            self.router.accumulated_spend = old_spend

    def test_8_pii_guardrail(self):
        """
        Verifies that the local DeBERTa-v3 PII Guardrail operates correctly:
        - Enabling MASK action replaces PII with MASK placeholders pre-call and post-call.
        - Enabling BLOCK action throws ValueError (PII policy violation) pre-call.
        """
        logger.info("--- Test 8: DeBERTa-v3 PII Guardrail ---")
        
        # Save original PII state
        old_pii_enabled = getattr(self.router, "pii_enabled", False)
        old_pii_action = getattr(self.router, "pii_action", "MASK")
        
        try:
            # 1. Test MASK Scenario
            self.router.pii_enabled = True
            self.router.pii_action = "MASK"
            
            messages_pii = [{"role": "user", "content": "My SSN is 123-45-6789"}]
            response_mask = asyncio.run(self.router.execute_chat_completion(
                model="primary-cluster",
                messages=messages_pii,
                max_tokens=100,
                mock_sandbox=True
            ))
            
            content_mask = response_mask["choices"][0]["message"]["content"]
            self.assertNotIn("123-45-6789", content_mask)
            self.assertTrue(any(tag in content_mask for tag in ["SOCIAL_SECURITY_NUMBER", "CREDIT_CARD_NUMBER", "BANK_ACCOUNT_NUMBER"]), f"No expected PII mask placeholder found in: {content_mask}")
            logger.info("PII Masking verification succeeded.")
            
            # 2. Test BLOCK Scenario
            self.router.pii_action = "BLOCK"
            with self.assertRaises(ValueError) as context:
                asyncio.run(self.router.execute_chat_completion(
                    model="primary-cluster",
                    messages=messages_pii,
                    max_tokens=100,
                    mock_sandbox=True
                ))
            
            self.assertIn("PII policy violation", str(context.exception))
            logger.info("PII Blocking verification succeeded.")
            
        finally:
            # Restore original state
            self.router.pii_enabled = old_pii_enabled
            self.router.pii_action = old_pii_action

    def test_9_deberta_model_training(self):
        """Verifies the DeBERTa model training and reload endpoints."""
        logger.info("--- Test 9: DeBERTa PII Model Training ---")
        base_url = "http://127.0.0.1:8090"
        
        # 1. Check training status is initially idle or completed
        status_resp = requests.get(f"{base_url}/v1/deberta/train/status")
        self.assertEqual(status_resp.status_code, 200)
        self.assertIn(status_resp.json()["status"], ["idle", "completed"])
        
        # 2. Trigger training with a very small 1-sample, 1-epoch payload
        payload = {
            "dataset": [
                {
                    "text": "My name is John Doe and my email is john@doe.com.",
                    "entities": [
                        {"start": 11, "end": 19, "label": "person"},
                        {"start": 36, "end": 48, "label": "email address"}
                    ]
                }
            ],
            "epochs": 1,
            "learning_rate": 5e-5,
            "batch_size": 1
        }
        
        train_resp = requests.post(f"{base_url}/v1/deberta/train", json=payload)
        self.assertEqual(train_resp.status_code, 200)
        self.assertEqual(train_resp.json()["status"], "training")
        
        # 3. Poll status until completed (timeout of 60 seconds)
        completed = False
        for _ in range(30):
            time.sleep(2.0)
            status_resp = requests.get(f"{base_url}/v1/deberta/train/status")
            status = status_resp.json()["status"]
            logger.info(f"Training status: {status} - Progress: {status_resp.json()['progress']}")
            if status == "completed":
                completed = True
                break
            elif status == "failed":
                self.fail(f"Training failed with error: {status_resp.json()['error']}")
                
        self.assertTrue(completed, "Training did not complete within the 60-second limit.")
        
        # 4. Verify local model files were generated
        import os
        current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        local_model_dir = os.path.join(current_dir, "models", "finetuned-deberta")
        self.assertTrue(os.path.exists(os.path.join(local_model_dir, "config.json")), "Local config.json was not generated.")
        logger.info("DeBERTa fine-tuning integration test passed successfully.")

    def test_x_pii_config_persistence(self):
        """Verifies PII config writes/reads across restarts and persistence to disk."""
        logger.info("--- Test X: PII Config Persistence ---")
        import json
        base_url = "http://127.0.0.1:8090"
        
        # Determine the config file path
        current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        config_path = os.path.join(current_dir, "config", "pii_guardrail_config.json")
        
        # Save backup of existing config if any
        config_backup = None
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_backup = f.read()
        
        try:
            # 1. Send POST to /ui/pii-config
            test_policy = {"person": "MASK", "email address": "BLOCK"}
            payload = {
                "pii_enabled": True,
                "pii_action": "BLOCK",
                "pii_policy": test_policy
            }
            
            resp = requests.post(f"{base_url}/ui/pii-config", json=payload)
            self.assertEqual(resp.status_code, 200)
            
            # 2. Assert that config/pii_guardrail_config.json is generated with the correct contents
            self.assertTrue(os.path.exists(config_path))
            with open(config_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data["pii_enabled"], True)
            self.assertEqual(saved_data["pii_action"], "BLOCK")
            self.assertEqual(saved_data["pii_policy"], test_policy)
            
            # 3. Instantiate a new LiteLLMProxyRouter and assert that it successfully reads the saved settings on startup
            new_router = LiteLLMProxyRouter(config=self.config)
            self.assertEqual(new_router.pii_enabled, True)
            self.assertEqual(new_router.pii_action, "BLOCK")
            self.assertEqual(new_router.pii_policy, test_policy)
            
            logger.info("PII config persistence verification succeeded.")
            
        finally:
            # Clean up: delete or restore config file
            if config_backup is not None:
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(config_backup)
            else:
                if os.path.exists(config_path):
                    try:
                        os.remove(config_path)
                    except Exception as e:
                        logger.warning(f"Could not remove test config path {config_path}: {e}")


    def test_z_custom_class_pipeline(self):
        """Verifies the complete end-to-end custom class pipeline."""
        logger.info("--- Test Z: Custom Class End-to-End Pipeline ---")
        base_url = "http://127.0.0.1:8090"
        
        # 1. Fetch current pii config and verify 'membership card' is not in initial labels
        pii_cfg_resp = requests.get(f"{base_url}/ui/pii-config")
        self.assertEqual(pii_cfg_resp.status_code, 200)
        initial_labels = pii_cfg_resp.json().get("active_labels", [])
        
        # Remove it if already there from previous dirty test runs
        if "membership card" in initial_labels:
            logger.info("Cleaning up 'membership card' config from config.json before testing...")
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "finetuned-deberta", "config.json")
            if os.path.exists(config_file):
                import json
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                id2label = cfg.get("id2label", {})
                new_id2lbl = {k: v for k, v in id2label.items() if "MEMBERSHIP_CARD" not in v}
                cfg["id2label"] = new_id2lbl
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                # clear cache
                from guardrails.deberta_pii_guardrail import _custom_labels_cache
                global _custom_labels_cache
                import guardrails.deberta_pii_guardrail as dpg
                dpg._custom_labels_cache = None
                
            pii_cfg_resp = requests.get(f"{base_url}/ui/pii-config")
            initial_labels = pii_cfg_resp.json().get("active_labels", [])
            
        self.assertNotIn("membership card", initial_labels)
        
        # 2. Trigger training with a custom class 'membership card' (labeled as 'Membership Card' in dataset)
        # Duplicate training sample to ensure weight convergence in 3 epochs
        single_sample = {
            "text": "The client's Membership Card is MEM-98765-XYZ",
            "entities": [
                {"start": 32, "end": 45, "label": "Membership Card"}
            ]
        }
        payload = {
            "dataset": [single_sample] * 10,
            "epochs": 3,
            "learning_rate": 1e-4,
            "batch_size": 1
        }
        train_resp = requests.post(f"{base_url}/v1/deberta/train", json=payload)
        self.assertEqual(train_resp.status_code, 200)
        
        # 3. Poll status until completed (timeout of 60 seconds)
        completed = False
        for _ in range(30):
            time.sleep(2.0)
            status_resp = requests.get(f"{base_url}/v1/deberta/train/status")
            status = status_resp.json()["status"]
            logger.info(f"Custom training status: {status} - Progress: {status_resp.json()['progress']}")
            if status == "completed":
                completed = True
                break
            elif status == "failed":
                self.fail(f"Custom training failed with error: {status_resp.json()['error']}")
                
        self.assertTrue(completed, "Custom training did not complete in time.")
        
        # 4. Verify 'membership card' is now returned as a custom active label in the config
        pii_cfg_resp = requests.get(f"{base_url}/ui/pii-config")
        self.assertEqual(pii_cfg_resp.status_code, 200)
        active_labels = pii_cfg_resp.json().get("active_labels", [])
        self.assertIn("membership card", active_labels)
        logger.info(f"Verified custom class 'membership card' is now in active labels: {active_labels}")
        
        # 5. Set policy for 'membership card' to MASK
        # Save original PII state first
        old_pii_enabled = getattr(self.router, "pii_enabled", False)
        old_pii_action = getattr(self.router, "pii_action", "MASK")
        old_pii_policy = getattr(self.router, "pii_policy", None)
        
        try:
            payload_config = {
                "pii_enabled": True,
                "pii_action": "MASK",
                "pii_policy": {
                    "membership card": "MASK"
                }
            }
            config_resp = requests.post(f"{base_url}/ui/pii-config", json=payload_config)
            self.assertEqual(config_resp.status_code, 200)
            
            # 6. Submit a completion request via HTTP containing Membership Card information
            # It should get masked to [MEMBERSHIP_CARD]
            payload_chat = {
                "model": "primary-cluster",
                "messages": [
                    {"role": "user", "content": "My Membership Card is MEM-98765-XYZ"}
                ],
                "max_tokens": 100,
                "mock_sandbox": True
            }
            completion_resp = requests.post(f"{base_url}/v1/chat/completions", json=payload_chat)
            self.assertEqual(completion_resp.status_code, 200)
            
            reply = completion_resp.json()["choices"][0]["message"]["content"]
            self.assertNotIn("MEM-98765-XYZ", reply)
            self.assertIn("[MEMBERSHIP_CARD]", reply)
            logger.info(f"Custom class MASK verification succeeded. Reply: {reply}")
            
            # 7. Set policy for 'membership card' to BLOCK
            payload_config["pii_policy"]["membership card"] = "BLOCK"
            config_resp = requests.post(f"{base_url}/ui/pii-config", json=payload_config)
            self.assertEqual(config_resp.status_code, 200)
            
            # 8. Submit prompt and verify it is BLOCKED
            completion_resp = requests.post(f"{base_url}/v1/chat/completions", json=payload_chat)
            self.assertEqual(completion_resp.status_code, 400)
            self.assertIn("PII policy violation", completion_resp.json()["detail"])
            logger.info("Custom class BLOCK verification succeeded.")
            
        finally:
            # Restore original state via POST config endpoint
            restore_payload = {
                "pii_enabled": old_pii_enabled,
                "pii_action": old_pii_action,
                "pii_policy": old_pii_policy
            }
            requests.post(f"{base_url}/ui/pii-config", json=restore_payload)

if __name__ == "__main__":
    unittest.main()

