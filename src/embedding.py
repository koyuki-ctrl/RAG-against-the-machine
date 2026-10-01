"""LangChain Embeddings backed by a local GGUF model (llama-cpp-python)."""
from __future__ import annotations

from typing import Any, List

from langchain_core.embeddings import Embeddings

from llama_cpp import Llama


from tqdm import tqdm


class ProgressEmbeddings(Embeddings):
    """Wrap any Embeddings and update a tqdm bar as batches are processed."""

    def __init__(
        self,
        inner: Embeddings,
        total: int,
        desc: str = "Embedding",
        unit: str = " chunk",
    ) -> None:
        self.inner = inner
        self.total = total
        self.desc = desc
        self.unit = unit
        self._bar: tqdm | None = None

    def _ensure_bar(self) -> tqdm:
        if self._bar is None:
            self._bar = tqdm(total=self.total, desc=self.desc, unit=self.unit)
        return self._bar

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        bar = self._ensure_bar()
        vectors = self.inner.embed_documents(texts)
        bar.update(len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)

    def close(self) -> None:
        """Close the bar when done (optional but clean)."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class GGUFEmbeddings(Embeddings):
    """Embedding model (e.g. all-MiniLM-L6-v2) exposed to LangChain."""

    def __init__(
        self,
        repo_id: str = "huoxu/all-MiniLM-L6-v2-Q8_0-GGUF",
        filename: str = "*q8_0.gguf",
        n_ctx: int = 512,
        n_threads: int | None = None,
        query_prefix: str = "task: search result | query: ",
        document_prefix: str = "title: none | text: ",
    ) -> None:
        """Store settings; the model is loaded lazily.

        Args:
            repo_id: Hugging Face repo of the GGUF model.
            filename: Glob selecting the GGUF file.
            n_ctx: Context window.
            n_threads: CPU threads (None = llama.cpp default).
            query_prefix: Prefix required by EmbeddingGemma for queries.
            document_prefix: Prefix required for documents.
        """
        self.repo_id = repo_id
        self.filename = filename
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model: Any = None

    def _load(self) -> Any:
        """Load the GGUF model in embedding mode (once)."""
        if self._model is None:
            self._model = Llama.from_pretrained(
                repo_id=self.repo_id,
                filename=self.filename,
                n_ctx=self.n_ctx,
                embedding=True,
                n_threads=self.n_threads,
                verbose=False,
            )
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents."""
        model = self._load()
        prefixed = [f"{self.document_prefix}{t}" for t in texts]
        out = model.create_embedding(prefixed)
        return [d["embedding"] for d in out["data"]]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        model = self._load()
        out = model.create_embedding(f"{self.query_prefix}{text}")
        return out["data"][0]["embedding"]
