from __future__ import annotations

import asyncio

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness

from src.config import get_settings
from src.retrieval.pipeline import RAGPipeline
from src.retrieval.retriever import HybridRetriever

TEST_CASES = [
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": "Retrieval-augmented generation combines retrieval over external knowledge with answer generation.",
    },
    {
        "question": "Why do teams use hybrid search in RAG?",
        "ground_truth": "Hybrid search combines keyword matching with dense semantic retrieval to improve recall and precision.",
    },
    {
        "question": "What is a reranker used for in a RAG pipeline?",
        "ground_truth": "A reranker reorders retrieved chunks so the most relevant passages are passed to the language model.",
    },
]


async def build_dataset() -> Dataset:
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    retriever = HybridRetriever(settings)

    rows = []
    for case in TEST_CASES:
        documents = retriever.retrieve(case["question"], top_k=settings.default_top_k)
        result = await pipeline.ask(case["question"], top_k=settings.default_top_k)
        rows.append(
            {
                "question": case["question"],
                "answer": result["answer"],
                "contexts": [document.page_content for document in documents],
                "ground_truth": case["ground_truth"],
            }
        )

    return Dataset.from_list(rows)


async def main() -> None:
    dataset = await build_dataset()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

