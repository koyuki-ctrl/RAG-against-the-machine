"""Pydantic data models exchanged between the RAG stages."""
from __future__ import annotations

import uuid
from typing import List

from pydantic import BaseModel, Field


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
