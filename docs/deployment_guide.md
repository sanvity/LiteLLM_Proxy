# LiteLLM Proxy Deployment Guide: Frontend (Vercel) & Backend (Render)

This guide details how to decouple and deploy the **LiteLLM Load-Balancing & Routing Proxy** application. We will deploy the FastAPI backend server to **Render** and the interactive dashboard frontend to **Vercel**.

---

## 1. Feasibility Analysis

Deploying this architecture is **highly feasible** and represents a standard, modern production setup. 

### Why this architecture is ideal:
* **Separation of Concerns:** The backend handles heavy operations (LiteLLM routing, token filtering with `tiktoken`, API retries, fallbacks, and backend metrics) while the frontend dashboard serves as a static UI.
* **Cost & Scaling:** Vercel serves the static HTML/JS assets instantly via its global CDN (Edge Network) for free. Render runs the persistent Python (FastAPI/Uvicorn) web service to process requests.
* **CORS Compatibility:** Since the backend is hosted on a different domain (`onrender.com`) than the frontend (`vercel.app`), browser security rules require CORS headers. Fortunately, the FastAPI application in [app.py](file:///c:/Users/soham/Downloads/litellm%20ki%20mkc/LiteLLM_Proxy-f39e1624a1ad0a568d17ba063799188b882327cc/proxy/app.py#L45-L52) already has CORSMiddleware enabled with `allow_origins=["*"]`.

### Core Deployment Strategies:
We propose two methods for connecting the Vercel frontend to the Render backend:

1. **Option A: Vercel Rewrites (Highly Recommended)**
   * We configure Vercel to reverse-proxy requests from `/metrics`, `/v1/chat/completions`, etc., directly to your Render backend domain.
   * **Pros:** Requires *zero* modifications to your frontend JavaScript file (keeps relative fetch requests like `fetch('/metrics')`). Bypasses CORS browser pre-flight checks entirely since, to the browser, the requests go to the same origin.
2. **Option B: Dynamic Base URL Configuration**
   * We update the JavaScript in [index.html](file:///c:/Users/soham/Downloads/litellm%20ki%20mkc/LiteLLM_Proxy-f39e1624a1ad0a568d17ba063799188b882327cc/proxy/templates/index.html) to call an absolute backend domain URL, configured via local storage or environment settings.
   * **Pros:** Simpler configuration if you want to swap backends dynamically.

---

## 2. Project Restructuring

By default, the backend FastAPI application serves the frontend template statically from the root route `/` in [app.py](file:///c:/Users/soham/Downloads/litellm%20ki%20mkc/LiteLLM_Proxy-f39e1624a1ad0a568d17ba063799188b882327cc/proxy/app.py#L60-L82). To split them, organize your repository so Vercel and Render can target their respective folders.

We recommend creating a `frontend` folder and keeping the backend in the root or a `backend` folder:

```text
LiteLLM_Proxy/
├── frontend/
│   ├── index.html       # Extracted frontend UI dashboard
│   └── vercel.json      # Vercel configuration (for URL rewrites)
├── proxy/               # FastAPI Python package
│   ├── app.py           # Backend routes (can keep or remove serve_ui route)
│   ├── router.py
│   └── config.py
├── config.yaml          # LiteLLM routing YAML config
├── requirements.txt     # Python requirements
├── main.py              # Backend entry point
└── .env                 # API credentials
```

---

## 3. Backend Deployment (Render)

Render is perfect for hosting the Python FastAPI web service.

### Step-by-Step Render Setup:
1. **Push your code** to a GitHub or GitLab repository.
2. Log in to [Render](https://render.com/) and click **New > Web Service**.
3. Connect your Git repository.
4. Configure the Web Service settings:
   * **Name:** `litellm-proxy-backend` (or similar)
   * **Environment:** `Python 3`
   * **Branch:** `main` (or your active branch)
   * **Root Directory:** `.` (or path to your backend folder if you nested it)
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `python main.py`
5. Click **Advanced** and configure **Environment Variables**:
   * Add any LLM provider keys you use:
     * `GROQ_API_KEY`: `your-groq-key`
     * `CEREBRAS_API_KEY`: `your-cerebras-key`
     * `TOGETHERAI_API_KEY`: `your-together-key`
     * `MISTRAL_API_KEY`: `your-mistral-key`
   * Render automatically injects `PORT` (usually 10000), which [main.py](file:///c:/Users/soham/Downloads/litellm%20ki%20mkc/LiteLLM_Proxy-f39e1624a1ad0a568d17ba063799188b882327cc/main.py#L23) will automatically read and bind to `0.0.0.0:$PORT`.
6. Configure **Health Check Path**:
   * Set the health check path to `/health`. The FastAPI app has a built-in health route in [app.py](file:///c:/Users/soham/Downloads/litellm%20ki%20mkc/LiteLLM_Proxy-f39e1624a1ad0a568d17ba063799188b882327cc/proxy/app.py#L84-L92) which returns `{"status": "healthy"}` and triggers deployment success.
7. Click **Create Web Service**.

> [!NOTE]
> Render's **Free Tier** web services spin down (go to sleep) after 15 minutes of inactivity. When a new request arrives, it will take 50 seconds to boot up (a cold start). If you need instant responses, upgrade the service to a **Paid Tier** (starting at $7/month).

---

## 4. Frontend Deployment (Vercel)

Vercel will host the `index.html` static file and proxy API calls to Render.

### Step-by-Step Vercel Setup:

#### 1. Extract the Frontend
Copy [index.html](file:///c:/Users/soham/Downloads/litellm%20ki%20mkc/LiteLLM_Proxy-f39e1624a1ad0a568d17ba063799188b882327cc/proxy/templates/index.html) out of `proxy/templates/` and place it inside a new `frontend/` folder in your project root. Rename it to `index.html` so Vercel can serve it as the default homepage.

#### 2. Create `vercel.json` for Rewrites (Option A)
Inside the `frontend/` folder, create a configuration file named `vercel.json` to handle routing proxy requests to Render:

```json
{
  "rewrites": [
    {
      "source": "/health",
      "destination": "https://your-backend-on-render.onrender.com/health"
    },
    {
      "source": "/metrics",
      "destination": "https://your-backend-on-render.onrender.com/metrics"
    },
    {
      "source": "/v1/:path*",
      "destination": "https://your-backend-on-render.onrender.com/v1/:path*"
    }
  ]
}
```
*Replace `https://your-backend-on-render.onrender.com` with the actual URL Render provides for your Web Service.*

#### 3. Deploy to Vercel
1. Log in to [Vercel](https://vercel.com/) and click **Add New > Project**.
2. Connect your Git repository.
3. Configure the Project settings:
   * **Framework Preset:** `Other` (or leave empty for static site)
   * **Root Directory:** Select `frontend` (this ensures Vercel only deploys the HTML/JS frontend assets).
4. Click **Deploy**.

Once deployment completes, Vercel will give you a domain (e.g., `https://litellm-proxy-ui.vercel.app`). Open this page in your browser; it will fetch metrics and send chat queries directly to Render without encountering CORS blocks!

---

## 5. Alternative Connection Method: Modifying JavaScript (Option B)

If you prefer to keep the codebase without a `vercel.json` configuration, modify the JS requests in the frontend file to target Render explicitly:

### 1. Update `frontend/index.html`
Modify the scripts in the frontend HTML to read a base URL. Locate the `fetch` calls and prefix them with `API_BASE_URL`:

```javascript
// At the beginning of the <script> block in index.html, define:
const API_BASE_URL = "https://your-backend-on-render.onrender.com";

// Update the metrics fetch:
const res = await fetch(`${API_BASE_URL}/metrics`);

// Update the models fetch:
const modRes = await fetch(`${API_BASE_URL}/v1/models`);

// Update the completions request:
const res = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
});
```

### 2. Tighten CORS on Render (Recommended)
Since the browser will now send cross-origin requests directly from Vercel to Render, restrict allowed origins in [app.py](file:///c:/Users/soham/Downloads/litellm%20ki%20mkc/LiteLLM_Proxy-f39e1624a1ad0a568d17ba063799188b882327cc/proxy/app.py#L46-L52) to secure your backend:

```python
self.app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "https://your-frontend-app.vercel.app"  # Your deployed Vercel domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Summary of Feasibility & Recommendations
* **Feasibility:** 10/10. It is extremely straightforward and highly recommended for scaling.
* **Best Route:** **Option A (Vercel Rewrites)** is recommended because you do not have to change a single line of fetching logic in the JS code and CORS is handled implicitly.
* **Security Note:** Make sure you restrict your `allow_origins` in CORSMiddleware once your Vercel URL is live, so that other websites cannot consume your proxy without authorization.
