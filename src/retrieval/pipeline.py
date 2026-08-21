from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings, openrouter_headers
from src.retrieval.retriever import HybridRetriever, unique_sources

PROMPT = ChatPromptTemplate.from_template(
    """
You are an AI research assistant.
Answer only from the provided context.
If the context is insufficient, say that clearly.
Use inline citations like [1] or [2] when you make a claim.
Keep the answer concise but concrete.

Context:
{context}

Question:
{question}
""".strip()
)


def build_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        title = document.metadata.get("title", "Unknown")
        url = document.metadata.get("url", "")
        published = document.metadata.get("published", "")
        source = document.metadata.get("source", "")
        blocks.append(
            "\n".join(
                [
                    f"[{index}] {title}",
                    f"URL: {url}",
                    f"Source: {source}",
                    f"Published: {published}",
                    f"Excerpt: {document.page_content}",
                ]
            )
        )
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.retriever = HybridRetriever(self.settings)
        self.llm = self._build_llm(self.settings.chat_model)

    def _build_llm(self, model: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            temperature=0,
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            default_headers=openrouter_headers(self.settings),
        )

    @staticmethod
    def _is_model_availability_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "model_not_found",
                "model is unavailable",
                "no endpoints found",
                "does not exist",
                "invalid model",
            )
        )

    async def ask(self, question: str, top_k: int | None = None) -> dict[str, object]:
        top_k = top_k or self.settings.default_top_k
        documents = self.retriever.retrieve(question, top_k=top_k)
        context = build_context(documents)
        models = (self.settings.chat_model, *self.settings.chat_fallback_models)
        answer: str | None = None
        last_error: Exception | None = None
        for index, model in enumerate(dict.fromkeys(models)):
            llm = self.llm if index == 0 else self._build_llm(model)
            chain = PROMPT | llm | StrOutputParser()
            try:
                answer = await chain.ainvoke({"context": context, "question": question})
                break
            except Exception as exc:
                last_error = exc
                if not self._is_model_availability_error(exc):
                    raise

        if answer is None:
            raise RuntimeError("No configured chat model is currently available.") from last_error
        return {
            "answer": answer,
            "sources": unique_sources(documents),
            "retrieved_chunks": len(documents),
        }
