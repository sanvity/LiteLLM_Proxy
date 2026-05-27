import os
import time
import requests
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
        response_short = self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages_short,
            max_tokens=100,
            mock_sandbox=True
        )
        model_short = response_short["model"]
        self.assertIn(model_short, [
            "groq/llama-3.1-8b-instant", 
            "cerebras/llama3.1-8b"
        ])
        logger.info(f"Short prompt routed correctly to primary node: {model_short}")
        
        # Case B: Huge Prompt (Exceeds Llama 3.1 8B's 131k context limit)
        messages_large = [{"role": "user", "content": "Explain gravity in great detail." * 1000}]
        response_large = self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages_large,
            max_tokens=130000,
            mock_sandbox=True
        )
        model_large = response_large["model"]
        self.assertEqual(model_large, "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        logger.info(f"Huge prompt auto-escalated correctly in sandbox to backup premium cluster: {model_large}")

    def test_4_mock_sandbox_completion(self):
        """Verifies that we can trigger the router completion flow in mock sandbox mode to test load balancing splits."""
        logger.info("--- Test 4: Mock Sandbox Load-Balancing Execution ---")
        
        messages = [{"role": "user", "content": "Compute the trajectory of a falling apple."}]
        
        response = self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages,
            max_tokens=100,
            mock_sandbox=True
        )
        
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
        """Tests that the custom local PII Shielding Engine correctly redacts names, SSNs, phone numbers, and emails."""
        logger.info("--- Test 6: Local PII Shielding Guardrail ---")
        
        vulnerable_prompt_input = '''import os
USER_FIRST_NAME = "Sanvi"
USER_LAST_NAME = "Jain"
USER_SSN = "456-45-6789"
USER_EMAIL = "sanvi.jain.private@gmail.com"
USER_PHONE = "+1-201-5550143"
print(f"Validating context block structures for user {USER_FIRST_NAME}")
give me the corrected version of this python code block structure'''

        sanitized_prompt = self.router.shield_prompt_payload(vulnerable_prompt_input)
        
        logger.info(f"Sanitized Prompt Output:\n{sanitized_prompt}")
        
        # Assert that all sensitive PII details are successfully masked
        self.assertNotIn("Sanvi", sanitized_prompt)
        self.assertNotIn("Jain", sanitized_prompt)
        self.assertNotIn("456-45-6789", sanitized_prompt)
        self.assertNotIn("sanvi.jain.private@gmail.com", sanitized_prompt)
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
        response_low = self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages_low,
            max_tokens=100,
            mock_sandbox=True
        )
        model_low = response_low["model"]
        # With low complexity prompt, the best node under primary-cluster is Cerebras Llama 3.1 8B (tier: low, cost: 0.01)
        self.assertEqual(model_low, "cerebras/llama3.1-8b")
        logger.info(f"Low complexity prompt routed correctly to low-cost node: {model_low}")
        
        # Case B: High Complexity prompt (Coding / optimization request)
        # Should match 'high' complexity and route to a high reasoning node (e.g. groq/llama-3.3-70b-versatile)
        messages_high = [{"role": "user", "content": "Optimize this SQL query and write a python refactoring function to execute it cleanly."}]
        response_high = self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages_high,
            max_tokens=150,
            mock_sandbox=True
        )
        model_high = response_high["model"]
        # With high complexity prompt, the best node under primary-cluster is Groq Llama 3.3 70B (tier: high, cost: 0.70)
        self.assertEqual(model_high, "groq/llama-3.3-70b-versatile")
        logger.info(f"High complexity prompt routed correctly to reasoning node: {model_high}")
        
        # Case C: Medium Complexity prompt (Summarization task)
        # Should match 'medium' complexity and route to a medium tier node (e.g. groq/llama-3.1-8b-instant)
        messages_med = [{"role": "user", "content": "Please write a concise summarization of this company's Q1 financial report."}]
        response_med = self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages_med,
            max_tokens=150,
            mock_sandbox=True
        )
        model_med = response_med["model"]
        # With medium complexity prompt, the best node under primary-cluster is Groq Llama 3.1 8B (tier: medium, cost: 0.05)
        self.assertEqual(model_med, "groq/llama-3.1-8b-instant")
        logger.info(f"Medium complexity prompt routed correctly to standard node: {model_med}")

    def test_8_reversible_pii_mapping(self):
        """Tests that the Reversible PII mapping anonymizes prompts and successfully restores original values in response."""
        logger.info("--- Test 8: Reversible PII Mapping Guardrail ---")
        
        raw_prompt = "My name is Sanvi Jain. My email is sanvi.jain.private@gmail.com. SSN is 456-45-6789. Phone is +1-201-5550143."
        
        # 1. Verify prompt sanitization separately to inspect placeholders and map
        sanitized_prompt, pii_map = self.router.shield_prompt_payload_reversible(raw_prompt)
        logger.info(f"Sanitized: {sanitized_prompt}")
        logger.info(f"PII Map: {pii_map}")
        
        self.assertNotIn("Sanvi", sanitized_prompt)
        self.assertNotIn("sanvi.jain.private@gmail.com", sanitized_prompt)
        self.assertIn("<PERSON_1>", sanitized_prompt)
        self.assertIn("<EMAIL_ADDRESS_1>", sanitized_prompt)
        self.assertEqual(pii_map["<PERSON_1>"], "Sanvi Jain")
        self.assertEqual(pii_map["<EMAIL_ADDRESS_1>"], "sanvi.jain.private@gmail.com")
        
        # 2. Execute chat completion with mock_sandbox=True to verify end-to-end de-anonymization restoration
        messages = [{"role": "user", "content": raw_prompt}]
        response = self.router.execute_chat_completion(
            model="primary-cluster",
            messages=messages,
            max_tokens=100,
            mock_sandbox=True
        )
        
        reply_content = response["choices"][0]["message"]["content"]
        logger.info(f"Restored Assistant Response:\n{reply_content}")
        
        # Original values must be fully restored in the final client response
        self.assertIn("Sanvi Jain", reply_content)
        self.assertIn("sanvi.jain.private@gmail.com", reply_content)
        self.assertIn("456-45-6789", reply_content)
        self.assertIn("+1-201-5550143", reply_content)
        
        # Placeholders must not remain in the user-returned response
        self.assertNotIn("<PERSON_1>", reply_content)
        self.assertNotIn("<EMAIL_ADDRESS_1>", reply_content)
        self.assertNotIn("<US_SSN_1>", reply_content)
        self.assertNotIn("<PHONE_NUMBER_1>", reply_content)
        
        logger.info("Reversible PII Mapping successfully validated end-to-end.")

if __name__ == "__main__":
    unittest.main()
