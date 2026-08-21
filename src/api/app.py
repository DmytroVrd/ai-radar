from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.api.schemas import HealthResponse, IndexResponse, QueryRequest, QueryResponse
from src.config import get_settings
from src.ingestion.fetcher import fetch_all_sources
from src.ingestion.indexer import collection_stats, index_articles, qdrant_is_available
from src.ingestion.scheduler import build_scheduler
from src.retrieval.pipeline import RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.pipeline = None
    app.state.scheduler = None

    if settings.scheduler_enabled:
        scheduler = build_scheduler(settings)
        scheduler.start()
        app.state.scheduler = scheduler

    yield

    scheduler = getattr(app.state, "scheduler", None)
    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(title="AI Radar", lifespan=lifespan)


def _pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        pipeline = RAGPipeline(request.app.state.settings)
        request.app.state.pipeline = pipeline
    return pipeline


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    status = "ok" if qdrant_is_available(settings=settings) else "degraded"
    status_label = "Ready" if status == "ok" else "Qdrant not ready"
    status_color = "#2f855a" if status == "ok" else "#c05621"
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Radar</title>
  <style>
    :root {{
      --bg: #f6f3ea;
      --panel: #fffdfa;
      --ink: #1f2933;
      --muted: #52606d;
      --line: #e7dfcf;
      --brand: #0f766e;
      --brand-dark: #115e59;
      --accent: #c05621;
      --ok: #2f855a;
      --shadow: 0 20px 50px rgba(31, 41, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Inter", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(192, 86, 33, 0.08), transparent 24%),
        var(--bg);
    }}
    .shell {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 40px 20px 56px;
    }}
    .hero {{
      display: grid;
      gap: 24px;
      grid-template-columns: 1.25fr 0.85fr;
      align-items: start;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .hero-copy {{
      padding: 32px;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.10);
      color: var(--brand-dark);
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    h1 {{
      margin: 18px 0 14px;
      font-size: clamp(40px, 7vw, 64px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .lead {{
      margin: 0;
      max-width: 62ch;
      font-size: 18px;
      line-height: 1.7;
      color: var(--muted);
    }}
    .cta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 24px;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      min-height: 48px;
      padding: 0 18px;
      border-radius: 14px;
      text-decoration: none;
      font-weight: 700;
      transition: 160ms ease;
    }}
    .button.primary {{
      background: var(--brand);
      color: white;
    }}
    .button.primary:hover {{ background: var(--brand-dark); }}
    .button.ghost {{
      color: var(--ink);
      border: 1px solid var(--line);
      background: white;
    }}
    .stats {{
      padding: 24px;
      display: grid;
      gap: 16px;
    }}
    .status-chip {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      width: fit-content;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(47, 133, 90, 0.10);
      color: {status_color};
      font-weight: 700;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: currentColor;
    }}
    .metric {{
      padding: 18px;
      border-radius: 18px;
      background: #fff;
      border: 1px solid var(--line);
    }}
    .metric-label {{
      font-size: 13px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric-value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 800;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin-top: 24px;
    }}
    .panel {{
      padding: 24px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 22px;
      letter-spacing: -0.03em;
    }}
    .muted {{
      color: var(--muted);
      line-height: 1.65;
      margin: 0;
    }}
    ul {{
      margin: 14px 0 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.7;
    }}
    .prompt-list {{
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }}
    .prompt {{
      padding: 14px 16px;
      border-radius: 16px;
      background: #fff;
      border: 1px solid var(--line);
      font-family: "Consolas", "SFMono-Regular", monospace;
      font-size: 14px;
      color: #234e52;
    }}
    .footer-note {{
      margin-top: 22px;
      color: var(--muted);
      font-size: 14px;
    }}
    code {{
      padding: 2px 6px;
      border-radius: 8px;
      background: rgba(15, 118, 110, 0.10);
      color: #134e4a;
      font-family: "Consolas", "SFMono-Regular", monospace;
    }}
    @media (max-width: 880px) {{
      .hero {{
        grid-template-columns: 1fr;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <article class="card hero-copy">
        <div class="eyebrow">AI Radar • Research Assistant</div>
        <h1>Track fresh AI ideas, then ask them questions.</h1>
        <p class="lead">
          AI Radar automatically collects recent AI engineering content from Hacker News, dev.to,
          arXiv, and curated AI blogs, indexes it in Qdrant, and answers questions with source links.
          This project is built to
          show a practical RAG pipeline: ingestion, hybrid retrieval, reranking, and cited answers.
        </p>
        <div class="cta-row">
          <a class="button primary" href="/docs">Open API Docs</a>
          <a class="button ghost" href="/health">Check Health</a>
          <a class="button ghost" href="/stats">View Stats</a>
        </div>
        <p class="footer-note">
          Start at <code>/docs</code> if you want to trigger <code>POST /index</code> and <code>POST /query</code>.
        </p>
      </article>

      <aside class="card stats">
        <div class="status-chip"><span class="dot"></span>{status_label}</div>
        <div class="metric">
          <div class="metric-label">API Status</div>
          <div class="metric-value">{status.upper()}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Indexed Collection</div>
          <div class="metric-value">ai_articles</div>
        </div>
        <div class="metric">
          <div class="metric-label">Data Sources</div>
          <div class="metric-value">HN · dev.to · arXiv · AI blogs</div>
        </div>
      </aside>
    </section>

    <section class="grid">
      <article class="card panel">
        <h2>What It Does</h2>
        <p class="muted">
          Instead of uploading PDFs manually, this assistant keeps a live research index of recent AI engineering writing.
          It is useful for summarizing trends, comparing ideas across sources, and tracing claims back to articles.
        </p>
        <ul>
          <li>Fetches fresh AI engineering articles automatically</li>
          <li>Indexes chunks into Qdrant</li>
          <li>Uses BM25 + vector search + reranking</li>
          <li>Returns answers with explicit sources</li>
        </ul>
      </article>

      <article class="card panel">
        <h2>How To Try It</h2>
        <p class="muted">
          The fastest workflow is simple: index articles once, then ask a question through the API docs.
        </p>
        <ul>
          <li>Open <code>/docs</code></li>
          <li>Run <code>POST /index</code></li>
          <li>Run <code>POST /query</code></li>
          <li>Read the answer and inspect source links</li>
        </ul>
      </article>

      <article class="card panel">
        <h2>Best Prompt Types</h2>
        <p class="muted">
          This project works best for recent-topic synthesis rather than general trivia.
        </p>
        <ul>
          <li>Recent RAG trends and techniques</li>
          <li>Hybrid search, reranking, chunking, agents</li>
          <li>Comparisons across recent sources</li>
          <li>AI tooling, open-source models, and infra updates</li>
        </ul>
      </article>
    </section>

    <section class="card panel" style="margin-top: 24px;">
      <h2>Example Questions</h2>
      <p class="muted">
        Try one of these in <code>POST /query</code> after indexing.
      </p>
      <div class="prompt-list">
        <div class="prompt">What changed in RAG recently?</div>
        <div class="prompt">Which recent sources discuss hybrid search or reranking?</div>
        <div class="prompt">Summarize recent trends in agentic AI tooling.</div>
        <div class="prompt">What new open-source model or AI tooling updates appeared recently?</div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.post("/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    try:
        result = await _pipeline(request).ask(payload.question, top_k=payload.top_k)
        return QueryResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/index", response_model=IndexResponse)
async def index_now(request: Request) -> IndexResponse:
    settings = request.app.state.settings
    if not qdrant_is_available(settings=settings):
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant is unavailable at {settings.qdrant_url}. Start it first.",
        )
    try:
        articles = await fetch_all_sources(settings=settings)
        report = index_articles(articles, settings=settings)
        return IndexResponse(**report.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
async def stats(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    if not qdrant_is_available(settings=settings):
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant is unavailable at {settings.qdrant_url}. Start it first.",
        )
    try:
        return collection_stats(settings=settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    if qdrant_is_available(settings=settings):
        return HealthResponse(status="ok")
    return HealthResponse(status="degraded")
