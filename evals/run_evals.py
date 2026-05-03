from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings, openrouter_headers  # noqa: E402
from src.embeddings import build_embeddings  # noqa: E402
from src.retrieval.pipeline import RAGPipeline  # noqa: E402
from src.retrieval.retriever import HybridRetriever  # noqa: E402

TEST_CASES = [
    {
        "question": "What changed in RAG recently?",
        "ground_truth": (
            "A good answer should summarize recent retrieval-augmented generation updates from the indexed sources, "
            "including retrieval methods, reranking, evaluation, tooling, or production tradeoffs when present."
        ),
    },
    {
        "question": "Which recent sources discuss hybrid search or reranking?",
        "ground_truth": (
            "A good answer should identify sources that mention hybrid search, keyword plus vector retrieval, "
            "reranking, or context selection improvements."
        ),
    },
    {
        "question": "What new open-source model or AI tooling updates appeared recently?",
        "ground_truth": (
            "A good answer should summarize recent open-source model, library, platform, or tooling updates "
            "found in the indexed AI engineering content."
        ),
    },
    {
        "question": "What are recent trends in agentic AI tooling?",
        "ground_truth": (
            "A good answer should discuss recent patterns around agents, tool use, workflows, orchestration, "
            "or developer tooling when those topics appear in the retrieved context."
        ),
    },
    {
        "question": "Which recent papers or posts mention LLM inference or model infrastructure?",
        "ground_truth": (
            "A good answer should identify sources about inference, serving, model infrastructure, latency, "
            "deployment, or related ML systems topics."
        ),
    },
    {
        "question": "What do recent sources say about embeddings or retrieval quality?",
        "ground_truth": (
            "A good answer should focus on embeddings, retrieval quality, ranking, search relevance, "
            "or evaluation signals present in the retrieved sources."
        ),
    },
    {
        "question": "What practical AI engineering lessons appear across the latest indexed articles?",
        "ground_truth": (
            "A good answer should synthesize applied engineering lessons such as evaluation, reliability, "
            "deployment, cost, latency, retrieval, or tool design from the indexed material."
        ),
    },
    {
        "question": "Which indexed sources discuss evaluation of AI systems?",
        "ground_truth": (
            "A good answer should mention sources about AI evaluation, benchmarks, eval bottlenecks, "
            "quality measurement, or model assessment when available."
        ),
    },
    {
        "question": "What are recent updates around LLM apps or developer tools?",
        "ground_truth": (
            "A good answer should summarize updates related to LLM application development, frameworks, "
            "developer workflows, SDKs, APIs, or applied AI tools."
        ),
    },
    {
        "question": "Compare recent discussion themes across arXiv and engineering blogs.",
        "ground_truth": (
            "A good answer should compare research-oriented themes from arXiv with practical engineering "
            "themes from blogs or developer-focused sources."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for AI Radar.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test cases. Useful for a quick smoke test.",
    )
    return parser.parse_args()


async def build_dataset(limit: int | None = None) -> Dataset:
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    retriever = HybridRetriever(settings)

    rows = []
    cases = TEST_CASES[:limit] if limit else TEST_CASES
    print(f"Building RAGAS dataset from {len(cases)} test cases...")
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] Retrieving and answering: {case['question']}")
        documents = retriever.retrieve(case["question"], top_k=settings.default_top_k)
        result = await pipeline.ask(case["question"], top_k=settings.default_top_k)
        rows.append(
            {
                "user_input": case["question"],
                "response": result["answer"],
                "retrieved_contexts": [document.page_content for document in documents],
                "reference": case["ground_truth"],
            }
        )

    print("Dataset is ready. Starting RAGAS metrics...")
    return Dataset.from_list(rows)


async def main(args: argparse.Namespace) -> None:
    settings = get_settings()
    dataset = await build_dataset(limit=args.limit)
    evaluator_llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers=openrouter_headers(settings),
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=evaluator_llm,
        embeddings=build_embeddings(settings),
    )
    print("RAGAS result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
