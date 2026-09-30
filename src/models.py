"""Pydantic data models exchanged between the RAG stages."""
from __future__ import annotations

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class MinimalSource(BaseModel):
    """A location in the corpus: file path + character range."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without ground truth."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results + generated answer for one question."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output of ``search_dataset``."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of ``answer_dataset``."""

    search_results: List[MinimalAnswer]
    k: int


class IndexedChunk(BaseModel):
    """A chunk stored in the index (internal model)."""

    source: str
    first_character_index: int
    last_character_index: int
    extension: str = ""
    breadcrumb: str = ""
    text: str


class SearchRequest(BaseModel):
    """Body of ``POST /search``."""

    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject blank queries and trim surrounding spaces."""
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class AnswerRequest(SearchRequest):
    """Body of ``POST /answer``."""

    k: int = Field(default=5, ge=1, le=50)


class HealthStatus(BaseModel):
    """Response of ``GET /health``."""

    status: str
    index_loaded: bool
    chunks: int
    index_version: Optional[str] = None
    cache_enabled: bool
    indexing: bool = False


class IndexRequest(BaseModel):
    """Body of ``POST /index`` (optional)."""

    max_chunk_size: int = Field(default=2000, ge=100, le=2000)


class UpdateRequest(BaseModel):
    """Body of ``POST /update`` (optional)."""

    max_chunk_size: Optional[int] = Field(default=None, ge=100, le=2000)


class IndexResult(BaseModel):
    """Response of ``POST /index``."""

    chunks: int
    index_version: str
    duration_seconds: float


class UpdateResult(IndexResult):
    """Response of ``POST /update``."""

    added: int
    modified: int
    deleted: int
    unchanged: int
    up_to_date: bool
