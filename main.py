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
    Spawns the Streamlit frontend as a background process on localhost only.
    Passes flags required for operation behind a reverse proxy.
    Streams Streamlit's stdout/stderr to the main process (visible in Railway/cloud logs).
    """
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    # Streamlit calls FastAPI internally on localhost
    env.setdefault("PROXY_URL", f"http://127.0.0.1:{os.environ.get('PORT', '8000')}")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", str(port),
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--server.baseUrlPath", "",
            "--browser.gatherUsageStats", "false",
        ],
        env=env,
        # Pipe output so errors appear in Railway/cloud platform logs
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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

        # Stream Streamlit's stdout to our logger (visible in Railway / cloud logs)
        import threading

        def _log_streamlit_output(proc: subprocess.Popen):
            streamlit_logger = logging.getLogger("streamlit")
            for line in iter(proc.stdout.readline, b""):
                try:
                    streamlit_logger.info(line.decode("utf-8", errors="replace").rstrip())
                except Exception:
                    pass

        log_thread = threading.Thread(target=_log_streamlit_output, args=(streamlit_proc,), daemon=True)
        log_thread.start()

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
        #    Timeout is 60s to accommodate Railway cold starts + DeBERTa load
        # ------------------------------------------------------------------
        streamlit_url = f"http://127.0.0.1:{streamlit_port}"
        logger.info(f"Waiting for Streamlit to be ready at {streamlit_url}...")
        ready = _wait_for_streamlit(streamlit_url, timeout=60)
        if ready:
            logger.info("✅ Streamlit is ready.")
        else:
            logger.warning(
                "⚠️  Streamlit did not respond within 60s. "
                "FastAPI will still start — Streamlit UI will auto-refresh when ready."
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
