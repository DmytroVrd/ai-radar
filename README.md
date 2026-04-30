# AI Radar

AI Radar is a RAG-powered research assistant that continuously ingests AI articles from Hacker News, dev.to, and arXiv, indexes them into Qdrant, and answers questions with explicit sources.

## Why this project stands out

- Automatic ingestion instead of manual PDF upload.
- Hybrid retrieval: dense vector search in Qdrant plus BM25 keyword search.
- Cross-encoder reranking to improve final context quality.
- RAGAS evaluation script so answer quality is measured, not guessed.
- FastAPI backend plus aiogram Telegram bot for a real user-facing interface.

## Architecture

```text
Telegram Bot (aiogram)
        |
        v
FastAPI  /query /index /stats /health
        |
        +--> Ingestion Scheduler
        |
        +--> RAG Pipeline
               |- BM25 retrieval over indexed chunks
               |- Qdrant dense vector search
               |- Reciprocal rank fusion
               |- Cross-encoder reranker
               |- OpenRouter-backed answer synthesis with citations
```

## Project layout

```text
src/
  api/
  bot/
  ingestion/
  retrieval/
evals/
tests/
```

## Quick start

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy the environment template and fill in the keys.

```bash
copy .env.example .env
```

3. Fill in the minimum required variables in `.env`.

```env
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
CHAT_MODEL=openrouter/free
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
```

4. Start Qdrant and the API.

```bash
docker compose up qdrant -d
python main.py
```

5. Index fresh articles.

```bash
curl -X POST http://localhost:8000/index
```

6. Ask a question.

```bash
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What changed in RAG over the last month?\",\"top_k\":5}"
```

7. Run the Telegram bot.

```bash
python -m src.bot.main
```

## Example prompts

- `What changed in RAG over the last month?`
- `Which arXiv papers mention retrieval latency tradeoffs?`
- `Summarize recent agent and tool-use patterns in LLM apps.`

## Evaluation

`evals/run_evals.py` runs a lightweight RAGAS evaluation over sample questions and prints metric scores for:

- `faithfulness`
- `answer_relevancy`
- `context_recall`

Those results are meant for the README, demos, and resume bullets once you have a few stable evaluation runs.

## API endpoints

- `POST /query`
- `POST /index`
- `GET /stats`
- `GET /health`

## Resume-friendly summary

Built a live RAG pipeline over AI articles from Hacker News, dev.to, and arXiv using FastAPI, Qdrant, LangChain, aiogram, cross-encoder reranking, RAGAS evaluation, and OpenRouter as the model gateway.
