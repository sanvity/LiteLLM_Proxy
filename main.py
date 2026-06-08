import os
import sys
import logging

import uvicorn
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("proxy.main")


def main():
    """
    Starts the LiteLLM Gateway FastAPI server.

    In production (Railway), this service runs independently.
    The Streamlit UI runs as a separate Railway service and connects
    to this API via the FASTAPI_URL environment variable.

    For local development, run both together with:
        python main.py          # This process (FastAPI on PORT)
        streamlit run app.py    # Separate terminal (Streamlit on 8501)
    """
    logger.info("Loading environment variables...")
    load_dotenv()

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting LiteLLM Gateway on {host}:{port}")

    try:
        from proxy import LiteLLMProxyApp

        proxy_service = LiteLLMProxyApp(config_path="config.yaml")
        app = proxy_service.get_app()

        logger.info(f"🚀 LiteLLM Gateway online → http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")

    except Exception as e:
        logger.critical(f"Failed to start LiteLLM Routing Proxy: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
