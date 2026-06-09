import os
import uvicorn
import logging
from dotenv import load_dotenv

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("proxy.main")

def main():
    # Load environment variables
    logger.info("Loading environment variables from .env...")
    load_dotenv()
    
    # Port and host configurations
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Initializing LiteLLMProxyApp on {host}:{port}...")
    
    try:
        from proxy import LiteLLMProxyApp
        
        # Instantiate our class-based proxy microservice
        proxy_service = LiteLLMProxyApp(config_path="config.yaml")
        app = proxy_service.get_app()
        
        logger.info("Starting production Uvicorn ASGI server...")
        uvicorn.run(app, host=host, port=port)
        
    except Exception as e:
        logger.critical(f"Failed to start the LiteLLM Routing Proxy Server: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
