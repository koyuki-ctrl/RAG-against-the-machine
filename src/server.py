"""Local HTTP API (FastAPI) to index, query and answer questions.

Run it with ``uv run python -m src serve`` or ``make run_server``.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException

from .cache import DiskCache, index_version
from .llm import LLM
from .models import (
    AnswerRequest,
    HealthStatus,
    IndexedChunk,
    IndexRequest,
    IndexResult,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    SearchRequest,
    UpdateRequest,
    UpdateResult,
)
from .utils import Retriever, Utils, build_context

NO_CONTEXT = "No relevant context found for this question."


def _sources(chunks: List[IndexedChunk]) -> List[MinimalSource]:
    """Convert chunks to the sources format used by the CLI."""
    return [
        MinimalSource(
            file_path=c.source,
            first_character_index=c.first_character_index,
            last_character_index=c.last_character_index,
        )
        for c in chunks
    ]


class RagService:
    """Index, LLM and locks shared by every request."""

    def __init__(
        self,
        index_dir: str,
        cache_dir: str,
        no_cache: bool,
        raw_dir: str,
    ) -> None:
        """Create the service (nothing heavy is loaded here).

        Args:
            index_dir: Directory of the BM25 index.
            cache_dir: Directory of the cache.
            no_cache: Disable every cache when True.
            raw_dir: Corpus directory used by ``/index`` and ``/update``.
        """
        self.index_dir = index_dir
        self.raw_dir = raw_dir
        self.cache: Optional[DiskCache] = (
            None if no_cache else DiskCache(cache_dir)
        )
        self.llm = LLM(cache=self.cache)
        self.llm_lock = threading.Lock()
        self.index_lock = threading.Lock()
        self._retriever: Optional[Retriever] = None
        self._load_lock = threading.Lock()

    @contextmanager
    def exclusive_job(self) -> Generator[None, None, None]:
        """Allow a single indexing job at a time.

        Raises:
            HTTPException: 409 if another job is already running.
        """
        if not self.index_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail="An indexing job is already running",
            )
        try:
            yield
        finally:
            self.index_lock.release()

    def retriever(self) -> Retriever:
        """Return an up-to-date retriever.

        The index is loaded once (fully in memory, so it can be rewritten
        on disk without affecting running searches), then reloaded
        automatically when ``index`` or ``update`` changed it.

        Returns:
            The retriever.

        Raises:
            HTTPException: 503 if the index is missing or unreadable.
        """
        path = Path(self.index_dir)
        with self._load_lock:
            if not (path / "chunks.json").is_file():
                self._retriever = None
                raise HTTPException(
                    status_code=503,
                    detail="No index found: run the 'index' command first",
                )
            try:
                current = index_version(path)
                if (
                    self._retriever is None
                    or self._retriever.version != current
                ):
                    self._retriever = Retriever(
                        self.index_dir, self.cache, mmap=False
                    )
            except Exception as exc:
                self._retriever = None
                raise HTTPException(
                    status_code=503,
                    detail=f"The index cannot be loaded: {exc}",
                ) from None
            return self._retriever


def create_app(
    index_dir: str = "data/processed",
    cache_dir: str = "data/cache",
    no_cache: bool = False,
    raw_dir: str = "data/raw",
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        index_dir: Directory of the BM25 index.
        cache_dir: Directory of the cache.
        no_cache: Disable every cache when True.
        raw_dir: Corpus directory used by ``/index`` and ``/update``.

    Returns:
        The configured application.
    """
    service = RagService(index_dir, cache_dir, no_cache, raw_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        """Load the index once at startup (the API starts without it)."""
        try:
            service.retriever()
        except HTTPException as exc:
            print(f"[WARN] {exc.detail}")
        yield

    app = FastAPI(
        title="RAG against the machine",
        description="Index the vLLM repository, query it, answer questions.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthStatus)
    def health() -> HealthStatus:
        """Report whether the index is loaded."""
        try:
            retriever: Optional[Retriever] = service.retriever()
        except HTTPException:
            retriever = None
        return HealthStatus(
            status="ok",
            index_loaded=retriever is not None,
            chunks=len(retriever.chunks) if retriever else 0,
            index_version=retriever.version if retriever else None,
            cache_enabled=service.cache is not None,
            indexing=service.index_lock.locked(),
        )

    @app.post("/index", response_model=IndexResult)
    def build_index(req: Optional[IndexRequest] = None) -> IndexResult:
        """Build the whole index from the corpus (can take a while).

        Searches keep being served from the previous index meanwhile.
        """
        size = req.max_chunk_size if req else 2000
        start = time.perf_counter()
        with service.exclusive_job():
            try:
                count = Utils().build_index(
                    service.raw_dir, service.index_dir, size
                )
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Indexing failed: {exc}"
                ) from None
            version = service.retriever().version
        return IndexResult(
            chunks=count,
            index_version=version,
            duration_seconds=round(time.perf_counter() - start, 2),
        )

    @app.post("/update", response_model=UpdateResult)
    def update_index(req: Optional[UpdateRequest] = None) -> UpdateResult:
        """Incrementally re-index only new, modified or deleted files."""
        size = req.max_chunk_size if req else None
        start = time.perf_counter()
        with service.exclusive_job():
            if not (Path(service.index_dir) / "manifest.json").is_file():
                raise HTTPException(
                    status_code=409,
                    detail="No index to update: call POST /index first",
                )
            try:
                stats = Utils().update_index(
                    service.raw_dir, service.index_dir, size
                )
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Update failed: {exc}"
                ) from None
            version = service.retriever().version
        return UpdateResult(
            **stats,
            index_version=version,
            duration_seconds=round(time.perf_counter() - start, 2),
            up_to_date=not (
                stats["added"] or stats["modified"] or stats["deleted"]
            ),
        )

    @app.post("/search", response_model=MinimalSearchResults)
    def search(req: SearchRequest) -> MinimalSearchResults:
        """Return the top-k sources for a question."""
        hits = service.retriever().search(req.query, req.k)
        return MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=req.query,
            retrieved_sources=_sources(hits),
        )

    @app.post("/answer", response_model=MinimalAnswer)
    def answer(req: AnswerRequest) -> MinimalAnswer:
        """Retrieve the top-k sources and generate an answer."""
        hits = service.retriever().search(req.query, req.k)
        if not hits:
            text = NO_CONTEXT
        else:
            context = build_context(
                [
                    (
                        f"{c.source} [{c.first_character_index}:"
                        f"{c.last_character_index}]",
                        c.text,
                    )
                    for c in hits
                ]
            )
            try:
                with service.llm_lock:
                    text = service.llm.ask_llm(
                        question=req.query, context=context
                    )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Answer generation failed: {exc}",
                ) from None
        return MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question=req.query,
            retrieved_sources=_sources(hits),
            answer=text,
        )

    @app.delete("/cache")
    def clear_cache() -> Dict[str, int]:
        """Delete every cached file."""
        removed = service.cache.clear() if service.cache else 0
        return {"removed": removed}

    return app


app = create_app(
    index_dir=os.environ.get("RAG_INDEX_DIR", "data/processed"),
    cache_dir=os.environ.get("RAG_CACHE_DIR", "data/cache"),
    no_cache=os.environ.get("RAG_NO_CACHE", "") == "1",
    raw_dir=os.environ.get("RAG_RAW_DIR", "data/raw"),
)
