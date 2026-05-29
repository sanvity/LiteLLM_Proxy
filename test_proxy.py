import os
import time
import requests
import asyncio
import unittest
import threading
import logging
from dotenv import load_dotenv

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
        # Inject mock unreachable Aporia credentials to bypass unconfigured checks
        # and test Presidio connection failure fallbacks natively
        os.environ.setdefault("APORIA_API_KEY_1", "mock-aporia-key-1")
        os.environ.setdefault("APORIA_API_BASE_1", "https://unreachable-aporia-api-base-1.xyz")
        os.environ.setdefault("APORIA_API_KEY_2", "mock-aporia-key-2")
        os.environ.setdefault("APORIA_API_BASE_2", "https://unreachable-aporia-api-base-2.xyz")
        
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
        self.assertEqual(model_large, "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
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

    def test_6_pii_shield_guardrail(self):
        """Tests that the local PII Shielding Engine correctly redacts names, SSNs, phone numbers, and emails."""
        logger.info("--- Test 6: Local PII Shielding Guardrail ---")
        
        vulnerable_prompt_input = '''import os
USER_FIRST_NAME = "Alice"
USER_LAST_NAME = "Smith"
USER_SSN = "456-45-6789"
USER_EMAIL = "alice.smith.private@gmail.com"
USER_PHONE = "+1-201-5550143"
print(f"Validating context block structures for user {USER_FIRST_NAME}")
give me the corrected version of this python code block structure'''

        sanitized_prompt = self.router.shield_prompt_payload(vulnerable_prompt_input)
        
        logger.info(f"Sanitized Prompt Output:\n{sanitized_prompt}")
        
        # Assert that all sensitive PII details are successfully masked
        self.assertNotIn("Alice", sanitized_prompt)
        self.assertNotIn("Smith", sanitized_prompt)
        self.assertNotIn("456-45-6789", sanitized_prompt)
        self.assertNotIn("alice.smith.private@gmail.com", sanitized_prompt)
        self.assertNotIn("+1-201-5550143", sanitized_prompt)
        
        # Assert that correct PII redaction labels exist (either via Presidio or Regex failover)
        self.assertTrue(
            "<PERSON>" in sanitized_prompt or "[PERSON]" in sanitized_prompt or 
            "PERSON" in sanitized_prompt
        )
        self.assertTrue(
            "<US_SSN>" in sanitized_prompt or "[US_SSN]" in sanitized_prompt or 
            "US_SSN" in sanitized_prompt
        )
        self.assertTrue(
            "<EMAIL_ADDRESS>" in sanitized_prompt or "[EMAIL_ADDRESS]" in sanitized_prompt or 
            "EMAIL" in sanitized_prompt
        )
        self.assertTrue(
            "<PHONE_NUMBER>" in sanitized_prompt or "[PHONE_NUMBER]" in sanitized_prompt or 
            "PHONE" in sanitized_prompt
        )
        
        logger.info("PII Shielding Guardrail successfully validated and confirmed.")

    def test_7_complexity_routing(self):
        """
        Tests that prompt complexity is correctly classified and routes requests
        to the appropriate complexity-tiered models.
        """
        logger.info("--- Test 7: Complexity-Aware Routing ---")
        
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

    def test_8_reversible_pii_mapping(self):
        """Tests that the Reversible PII mapping anonymizes prompts and successfully restores original values in response."""
        logger.info("--- Test 8: Reversible PII Mapping Guardrail ---")
        
        raw_prompt = "My name is Sanvi Jain. My email is sanvi.jain.private@gmail.com. SSN is 456-45-6789. Phone is +1-201-5550143. Card is 4111-1111-1111-1111."
        
        # 1. Verify prompt sanitization separately to inspect placeholders and map
        sanitized_prompt, pii_map = self.router.shield_prompt_payload_reversible(raw_prompt)
        logger.info(f"Sanitized: {sanitized_prompt}")
        logger.info(f"PII Map: {pii_map}")
        
        self.assertNotIn("Sanvi", sanitized_prompt)
        self.assertNotIn("sanvi.jain.private@gmail.com", sanitized_prompt)
        self.assertNotIn("4111-1111-1111-1111", sanitized_prompt)
        self.assertIn("<PERSON_1>", sanitized_prompt)
        self.assertIn("<EMAIL_ADDRESS_1>", sanitized_prompt)
        self.assertIn("<CREDIT_CARD_1>", sanitized_prompt)
        self.assertEqual(pii_map["<PERSON_1>"], "Sanvi Jain")
        self.assertEqual(pii_map["<EMAIL_ADDRESS_1>"], "sanvi.jain.private@gmail.com")
        self.assertEqual(pii_map["<CREDIT_CARD_1>"], "4111-1111-1111-1111")
        
        # 2. Execute chat completion with mock_sandbox=True to verify end-to-end de-anonymization restoration
        # Temporarily isolate to only Presidio PII guardrail
        old_instances = self.router.guardrail_instances
        self.router.guardrail_instances = {
            "presidio-pii": old_instances.get("presidio-pii", [])
        }
        
        try:
            messages = [{"role": "user", "content": raw_prompt}]
            response = asyncio.run(self.router.execute_chat_completion(
                model="primary-cluster",
                messages=messages,
                max_tokens=100,
                mock_sandbox=True
            ))
            
            reply_content = response["choices"][0]["message"]["content"]
            logger.info(f"Restored Assistant Response:\n{reply_content}")
            
            # Original values must be fully restored in the final client response
            self.assertIn("Sanvi Jain", reply_content)
            self.assertIn("sanvi.jain.private@gmail.com", reply_content)
            self.assertIn("456-45-6789", reply_content)
            self.assertIn("+1-201-5550143", reply_content)
            self.assertIn("4111-1111-1111-1111", reply_content)
            
            # Placeholders must not remain in the user-returned response
            self.assertNotIn("<PERSON_1>", reply_content)
            self.assertNotIn("<EMAIL_ADDRESS_1>", reply_content)
            self.assertNotIn("<US_SSN_1>", reply_content)
            self.assertNotIn("<PHONE_NUMBER_1>", reply_content)
            self.assertNotIn("<CREDIT_CARD_1>", reply_content)
            
        finally:
            self.router.guardrail_instances = old_instances
            
        logger.info("Reversible PII Mapping successfully validated end-to-end.")

    def test_9_advanced_guardrails(self):
        """
        Tests the advanced guardrails including Content Filter, Team-based BYO registration/approvals,
        Guardrail Load Balancing, the Testing Playground endpoint, and Realtime API transcriptions.
        """
        logger.info("--- Test 9: Advanced Guardrails Verification ---")
        
        # 1. Start uvicorn on port 8095 in a background thread to prevent collision
        os.environ["PORT"] = "8095"
        proxy_app = LiteLLMProxyApp(config_path="config.yaml")
        app_instance = proxy_app.get_app()
        
        import uvicorn
        server_thread = threading.Thread(
            target=lambda: uvicorn.run(app_instance, host="127.0.0.1", port=8095, log_level="warning"),
            daemon=True
        )
        server_thread.start()
        
        # Wait a moment for server boot
        time.sleep(1.5)
        
        base_url = "http://127.0.0.1:8095"
        
        # A. Verify Content Filter blocking (dynamic blocked keyword)
        payload_block = {
            "model": "primary-cluster",
            "messages": [
                {"role": "user", "content": "This is a toxic phrase."}
            ],
            "max_tokens": 100,
            "mock_sandbox": True
        }
        resp_block = requests.post(f"{base_url}/v1/chat/completions", json=payload_block)
        self.assertEqual(resp_block.status_code, 400)
        self.assertIn("Request blocked by content filter guardrail", resp_block.json()["detail"])
        logger.info("Content Filter BLOCK verified successfully via HTTP completion.")
        
        # B. Verify Content Filter masking (sensitive data)
        payload_mask = {
            "model": "primary-cluster",
            "messages": [
                {"role": "user", "content": "Here is some sensitive data for you."}
            ],
            "max_tokens": 100,
            "mock_sandbox": True
        }
        resp_mask = requests.post(f"{base_url}/v1/chat/completions", json=payload_mask)
        self.assertEqual(resp_mask.status_code, 200)
        reply_mask = resp_mask.json()["choices"][0]["message"]["content"]
        self.assertNotIn("sensitive data", reply_mask)
        self.assertTrue("<HARMFUL-CONTENT-FILTER>" in reply_mask or "HARMFUL-CONTENT-FILTER" in reply_mask)
        logger.info(f"Content Filter MASK verified successfully. Reply: {reply_mask}")
        
        # C. Verify Team Guardrails Register
        register_payload = {
            "guardrail_name": "team-custom-guard",
            "litellm_params": {
                "guardrail": "aporia",
                "mode": "pre_call",
                "api_base": "https://api.aporia.com/v1",
                "api_key": "aporia-secret-key-xyz"
            },
            "guardrail_info": {
                "owner": "team-ai-safety",
                "description": "Custom pre-call validation guardrail"
            }
        }
        resp_reg = requests.post(f"{base_url}/guardrails/register", json=register_payload)
        self.assertEqual(resp_reg.status_code, 200)
        reg_data = resp_reg.json()
        self.assertEqual(reg_data["guardrail_name"], "team-custom-guard")
        self.assertEqual(reg_data["status"], "pending_review")
        guardrail_id = reg_data["guardrail_id"]
        self.assertIsNotNone(guardrail_id)
        logger.info(f"Team Guardrail registered: ID={guardrail_id}")
        
        # Verify submission exists in list
        resp_list = requests.get(f"{base_url}/guardrails/submissions")
        self.assertEqual(resp_list.status_code, 200)
        submissions = resp_list.json()["submissions"]
        self.assertTrue(any(s["guardrail_id"] == guardrail_id for s in submissions))
        logger.info("Team submissions list audited successfully.")
        
        # D. Verify Approve Submission
        resp_appr = requests.post(f"{base_url}/guardrails/submissions/{guardrail_id}/approve")
        self.assertEqual(resp_appr.status_code, 200)
        self.assertEqual(resp_appr.json()["status"], "active")
        logger.info("Team Guardrail approved successfully.")
        
        # E. Verify Reject Submission (Register a second one and reject it)
        register_payload2 = {
            "guardrail_name": "team-rejected-guard",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "blocked_words": [{"keyword": "toxic phrase", "action": "BLOCK"}]
            }
        }
        resp_reg2 = requests.post(f"{base_url}/guardrails/register", json=register_payload2)
        guardrail_id2 = resp_reg2.json()["guardrail_id"]
        
        resp_rej = requests.post(f"{base_url}/guardrails/submissions/{guardrail_id2}/reject")
        self.assertEqual(resp_rej.status_code, 200)
        self.assertEqual(resp_rej.json()["status"], "rejected")
        logger.info("Team Guardrail rejection verified successfully.")
        
        # F. Verify Test Playground Endpoint
        test_playground_payload = {
            "text": "This has a toxic phrase and also USER_FIRST_NAME is Alice Smith.",
            "guardrails": ["harmful-content-filter", "aporia-pre-guard"]
        }
        resp_play = requests.post(f"{base_url}/guardrails/test", json=test_playground_payload)
        self.assertEqual(resp_play.status_code, 200)
        play_data = resp_play.json()
        self.assertIn("results", play_data)
        results = play_data["results"]
        # Ensure harmful-content-filter blocked the toxic phrase, and aporia-pre-guard masked Alice Smith
        content_filter_res = next((r for r in results if r["guardrail_name"] == "harmful-content-filter"), None)
        aporia_res = next((r for r in results if r["guardrail_name"] == "aporia-pre-guard"), None)
        
        self.assertIsNotNone(content_filter_res)
        self.assertIsNotNone(aporia_res)
        
        self.assertEqual(content_filter_res["action"], "BLOCK")
        self.assertEqual(content_filter_res["passed"], False)
        
        self.assertEqual(aporia_res["action"], "MASK")
        self.assertEqual(aporia_res["passed"], False)
        self.assertNotIn("Alice Smith", aporia_res["output"])
        
        logger.info(f"Guardrail Test Playground verified successfully. Results: {results}")
        
        # G. Verify Realtime Audio Transcription turn validation
        # 1) Toxic phrase should block
        res_rt_block = asyncio.run(self.router.validate_realtime_transcription("This contains a toxic phrase", guardrail_name="harmful-content-filter"))
        self.assertEqual(res_rt_block["action"], "BLOCK")
        self.assertEqual(res_rt_block["guardrail"], "harmful-content-filter")
        
        # 2) Regular phrase should pass (ALLOW)
        res_rt_allow = asyncio.run(self.router.validate_realtime_transcription("Hello, can you explain the trajectory of falling gravity?"))
        self.assertEqual(res_rt_allow["action"], "ALLOW")
        self.assertIsNone(res_rt_allow["guardrail"])
        
        logger.info("Realtime API Audio transcription turn validation verified successfully.")

    def test_10_aporia_ssn_aadhaar(self):
        """
        Tests Aporia guardrail simulation and connection enforcement:
        1. Strict no-fallback/fail-loud behavior if credentials are not configured.
        2. Injecting configuration, simulating a connection timeout, and verifying
           that it triggers the fallback to Presidio to redact sensitive PII (PERSON, US_SSN, EMAIL, PHONE).
        """
        logger.info("--- Test 10: Aporia PII & Connection Enforcement ---")
        
        # 1. Retrieve a load-balanced instance of the pre-configured aporia guardrail
        aporia_guard = self.router.get_guardrail_instance("aporia-pre-guard")
        self.assertIsNotNone(aporia_guard, "aporia-pre-guard is not registered in the router!")
        
        # Save original params
        old_key = aporia_guard.litellm_params.get("api_key")
        old_base = aporia_guard.litellm_params.get("api_base")
        
        # Test Case A: Strict 'fail loud' unconfigured check
        # Temporarily clear key/base
        aporia_guard.litellm_params["api_key"] = None
        aporia_guard.litellm_params["api_base"] = None
        
        try:
            with self.assertRaises(ValueError) as context:
                asyncio.run(aporia_guard.check_text("Some sensitive info check"))
            self.assertIn("Enterprise Aporia Guardrail is unconfigured", str(context.exception))
            logger.info("Strict unconfigured ValueError check passed.")
        finally:
            # Restore configuration to mock values for connection test
            aporia_guard.litellm_params["api_key"] = "mock-aporia-key"
            aporia_guard.litellm_params["api_base"] = "https://mock-unreachable-aporia-api.xyz"
            
        # Test Case B: Connection failure fallback to Presidio
        # We check text containing sensitive PII (Person, SSN, Phone, Email, Aadhaar)
        prompt_with_pii = "Contact Sanvi Jain at sanvi.jain@example.com or +1-201-5550143. SSN is 456-45-6789. Send Aadhaar card 2345 6789 1234."
        
        # Call check_text: it should try to connect to the unreachable domain,
        # fail due to connection error/timeout, catch the error, and execute Presidio fallback.
        is_blocked, action, reason = asyncio.run(aporia_guard.check_text(prompt_with_pii))
        
        self.assertTrue(is_blocked, "PII prompt should be flagged/blocked by Presidio fallback")
        self.assertEqual(action, "MASK")
        self.assertIn("[Presidio Fallback]", reason)
        
        # Anonymize using the fallback result stored in last revised text
        masked_text = aporia_guard.mask_text(prompt_with_pii)
        logger.info(f"Masked text after Presidio fallback: {masked_text}")
        
        # Assert that raw PII details are successfully masked
        self.assertNotIn("Sanvi", masked_text)
        self.assertNotIn("Jain", masked_text)
        self.assertNotIn("sanvi.jain@example.com", masked_text)
        self.assertNotIn("+1-201-5550143", masked_text)
        self.assertNotIn("456-45-6789", masked_text)
        self.assertNotIn("2345 6789 1234", masked_text)
        
        # Assert fallback labels exist (PERSON, EMAIL, PHONE, US_SSN, IDENTIFIER)
        self.assertTrue("<PERSON" in masked_text or "PERSON" in masked_text)
        self.assertTrue("<EMAIL" in masked_text or "EMAIL" in masked_text)
        self.assertTrue("<PHONE" in masked_text or "PHONE" in masked_text)
        self.assertTrue("<US_SSN" in masked_text or "US_SSN" in masked_text)
        self.assertTrue("<IDENTIFIER" in masked_text or "IDENTIFIER" in masked_text)
        
        # Restore original credentials at the end of the test
        if old_key:
            aporia_guard.litellm_params["api_key"] = old_key
        else:
            aporia_guard.litellm_params.pop("api_key", None)
        if old_base:
            aporia_guard.litellm_params["api_base"] = old_base
        else:
            aporia_guard.litellm_params.pop("api_base", None)
            
        logger.info("Aporia Strict configuration and Presidio connection-failure fallback validated successfully.")

    def test_11_aporia_circuit_breaker(self):
        """
        Tests the in-memory Circuit Breaker mechanism inside the Aporia client:
        - Simulates consecutive network/timeout failures.
        - Verifies that after 3 failures, the circuit breaker trips to Open (aporia_healthy = False).
        - Verifies that subsequent requests immediately fast-fail to Presidio without network attempts.
        """
        logger.info("--- Test 11: Aporia Circuit Breaker Tripping & Fast-Fail ---")
        
        aporia_guard = self.router.get_guardrail_instance("aporia-pre-guard")
        self.assertIsNotNone(aporia_guard)
        
        # Save original states
        old_healthy = aporia_guard.aporia_healthy
        old_failures = aporia_guard.consecutive_failures
        old_key = aporia_guard.litellm_params.get("api_key")
        old_base = aporia_guard.litellm_params.get("api_base")
        
        # Set to unreachable and reset breaker
        aporia_guard.aporia_healthy = True
        aporia_guard.consecutive_failures = 0
        aporia_guard.litellm_params["api_key"] = "mock-key"
        aporia_guard.litellm_params["api_base"] = "https://unreachable-domain-cb-test.xyz"
        
        try:
            # 1. Fire first failure
            asyncio.run(aporia_guard.check_text("Query 1"))
            self.assertEqual(aporia_guard.consecutive_failures, 1)
            self.assertTrue(aporia_guard.aporia_healthy)
            
            # 2. Fire second failure
            asyncio.run(aporia_guard.check_text("Query 2"))
            self.assertEqual(aporia_guard.consecutive_failures, 2)
            self.assertTrue(aporia_guard.aporia_healthy)
            
            # 3. Fire third failure -> Should trip the circuit
            asyncio.run(aporia_guard.check_text("Query 3"))
            self.assertEqual(aporia_guard.consecutive_failures, 3)
            self.assertFalse(aporia_guard.aporia_healthy) # Circuit is tripped/open!
            
            # 4. Fire subsequent query containing PII -> Should fast-fail to Presidio without hitting network
            # To prove it didn't hit network, we can change the URL to something bad and it should still pass cleanly
            aporia_guard.litellm_params["api_base"] = "https://another-garbage-url.invalid"
            is_blocked, action, reason = asyncio.run(aporia_guard.check_text("Send SSN 123-45-6789"))
            self.assertTrue(is_blocked)
            self.assertEqual(action, "MASK")
            self.assertIn("[Aporia Circuit Breaker] Circuit is OPEN", reason)
            
            logger.info("Circuit Breaker Tripping and Fast-Fail successfully validated.")
            
        finally:
            # Restore states
            aporia_guard.aporia_healthy = old_healthy
            aporia_guard.consecutive_failures = old_failures
            aporia_guard.litellm_params["api_key"] = old_key
            aporia_guard.litellm_params["api_base"] = old_base

    def test_12_aporia_presidio_dual_failure_fail_closed(self):
        """
        Tests the Fail-Closed security strategy when both Aporia and local Presidio are unreachable/unhealthy:
        - Simulates a tripped/failed Aporia state.
        - Mocks a Presidio failure (throws exception).
        - Verifies that the system throws a loud ValueError explaining both failed (Fail-Closed).
        """
        logger.info("--- Test 12: Fail-Closed Dual Failure Policy ---")
        
        aporia_guard = self.router.get_guardrail_instance("aporia-pre-guard")
        self.assertIsNotNone(aporia_guard)
        
        # Save original states
        old_healthy = aporia_guard.aporia_healthy
        old_failures = aporia_guard.consecutive_failures
        old_key = aporia_guard.litellm_params.get("api_key")
        old_base = aporia_guard.litellm_params.get("api_base")
        old_callback = self.router.pii_guardrail_callback
        
        # Force tripped state
        aporia_guard.aporia_healthy = False
        aporia_guard.last_failure_time = time.time()
        aporia_guard.litellm_params["api_key"] = "mock-key"
        aporia_guard.litellm_params["api_base"] = "https://unreachable-domain-dual-test.xyz"
        
        # Mock Presidio callback throwing exception
        class BrokenPresidio:
            def shield_text(self, text, metadata):
                raise RuntimeError("Presidio Analyzer is broken / model file missing!")
        
        self.router.pii_guardrail_callback = BrokenPresidio()
        
        try:
            with self.assertRaises(ValueError) as context:
                asyncio.run(aporia_guard.check_text("SSN is 123-45-6789"))
            
            self.assertIn("Fail-Closed: Critical safety threat", str(context.exception))
            logger.info("Dual failure Fail-Closed safety check passed.")
            
        finally:
            # Restore states
            aporia_guard.aporia_healthy = old_healthy
            aporia_guard.consecutive_failures = old_failures
            aporia_guard.litellm_params["api_key"] = old_key
            aporia_guard.litellm_params["api_base"] = old_base
            self.router.pii_guardrail_callback = old_callback

    def test_13_priority_preference_routing(self):
        """
        Tests the Priority-Based Preference Routing and Credit Limit Failover:
        - Configures custom preference order: [groq/llama-3.1-8b-instant, cerebras/llama3.1-8b]
        - Sets tight budget limits.
        - Verifies it routes requests to the first preferred model.
        - Exceeds the first model's budget, then verifies it automatically cascades/fails over to the second!
        - Resets the spend, verifying it goes back to the first.
        """
        logger.info("--- Test 13: Priority-Based Preference Routing & Credit Limits ---")
        
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

if __name__ == "__main__":
    unittest.main()

