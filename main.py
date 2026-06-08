import os
import sys
import time
import atexit
import subprocess
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


def _wait_for_streamlit(url: str, timeout: int = 30) -> bool:
    """
    Polls the Streamlit health endpoint until it responds or the timeout expires.
    Returns True if Streamlit is ready, False if it timed out.
    """
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _spawn_streamlit(app_path: str, port: int = 8501) -> subprocess.Popen:
    """
    Spawns the Streamlit frontend as a background process.
    Passes flags required for operation behind a reverse proxy:
      --server.enableCORS=false        — CORS handled by FastAPI
      --server.enableXsrfProtection=false — XSRF not needed behind proxy
      --server.baseUrlPath=""          — proxy serves from root
    """
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(port)
    # Tell Streamlit where the proxy API lives (Streamlit calls FastAPI)
    env.setdefault("PROXY_URL", f"http://127.0.0.1:{os.environ.get('PORT', '8000')}")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--server.baseUrlPath", "",
            "--browser.gatherUsageStats", "false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return proc


def main():
    # Load environment variables from .env (local dev) or injected cloud secrets
    logger.info("Loading environment variables...")
    load_dotenv()

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    streamlit_port = int(os.environ.get("STREAMLIT_PORT", 8501))

    logger.info(f"Starting LiteLLM Gateway on {host}:{port}")

    try:
        from proxy import LiteLLMProxyApp

        # ------------------------------------------------------------------
        # 1. Spawn Streamlit UI in the background
        # ------------------------------------------------------------------
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(current_dir, "app.py")

        logger.info(f"Spawning Streamlit UI process on internal port {streamlit_port}...")
        streamlit_proc = _spawn_streamlit(app_path, port=streamlit_port)

        def _cleanup():
            logger.info("Shutting down Streamlit process...")
            streamlit_proc.terminate()
            try:
                streamlit_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                streamlit_proc.kill()

        atexit.register(_cleanup)

        # ------------------------------------------------------------------
        # 2. Wait for Streamlit to be ready before accepting traffic
        #    (avoids 503 flash on first browser visit)
        # ------------------------------------------------------------------
        streamlit_url = f"http://127.0.0.1:{streamlit_port}"
        logger.info(f"Waiting for Streamlit to be ready at {streamlit_url}...")
        ready = _wait_for_streamlit(streamlit_url, timeout=40)
        if ready:
            logger.info("✅ Streamlit is ready.")
        else:
            logger.warning(
                "⚠️  Streamlit did not respond within 40s. "
                "FastAPI will still start — Streamlit UI will show a loading screen until ready."
            )

        # ------------------------------------------------------------------
        # 3. Start FastAPI / Uvicorn (this blocks until shutdown)
        # ------------------------------------------------------------------
        proxy_service = LiteLLMProxyApp(config_path="config.yaml")
        app = proxy_service.get_app()

        logger.info(f"🚀 LiteLLM Gateway online → http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")

    except Exception as e:
        logger.critical(f"Failed to start LiteLLM Routing Proxy: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
