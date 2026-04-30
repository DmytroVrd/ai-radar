from src.ingestion.indexer import stable_chunk_id


def test_stable_chunk_id_is_deterministic() -> None:
    first = stable_chunk_id("https://example.com/article", 0, "hello world")
    second = stable_chunk_id("https://example.com/article", 0, "hello world")

    assert first == second
    assert len(first) == 36
