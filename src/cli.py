"""Command-line interface (Python Fire) of the RAG system."""
from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, ParamSpec, Tuple

from pydantic import BaseModel
from tqdm import tqdm

from .models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .utils import Retriever, Utils

P = ParamSpec("P")

MAX_CONTEXT_CHARS = 9000
IOU_THRESHOLD = 0.05


def _safe(func: Callable[P, None]) -> Callable[P, None]:
    """Turn any exception into a clean error message (no traceback)."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            print("[ERROR] Interrupted", file=sys.stderr)
            raise SystemExit(130) from None
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from None

    return wrapper


def _check_k(k: Any) -> int:
    """Validate the top-k argument."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return k


def _check_query(query: Any) -> str:
    """Validate a query (Fire may parse '123' as an int)."""
    if query is None or not str(query).strip():
        raise ValueError("The query is empty")
    return str(query)


def _check_json(path: Any, name: str) -> str:
    """Validate a path to an existing .json file."""
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name} must be a non-empty path")
    if not path.endswith(".json"):
        raise ValueError(f"{name}: only .json files are accepted")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name}: file not found: {path}")
    return path


def _check_dir(path: Any, name: str) -> str:
    """Validate an output directory argument."""
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name} must be a non-empty path")
    return path


def _save(model: BaseModel, directory: str, basename: str) -> Path:
    """Write a pydantic model as JSON in ``directory``."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / basename
    out_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def _build_context(blocks: List[Tuple[str, str]]) -> str:
    """Format snippets for the prompt within a character budget."""
    parts: List[str] = []
    used = 0
    for i, (label, text) in enumerate(blocks, 1):
        room = MAX_CONTEXT_CHARS - used
        if room <= 200:
            break
        snippet = text[:room]
        parts.append(f"[{i}] {label}\n{snippet}")
        used += len(snippet)
    return "\n\n---\n\n".join(parts)


def _iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection over union of two character ranges."""
    inter = min(a.last_character_index, b.last_character_index) - max(
        a.first_character_index, b.first_character_index
    )
    if inter <= 0:
        return 0.0
    union = (
        (a.last_character_index - a.first_character_index)
        + (b.last_character_index - b.first_character_index)
        - inter
    )
    return inter / union if union > 0 else 0.0


class Arguments:
    """RAG commands: index, search, search_dataset, answer, ..."""

    @_safe
    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        index_dir: str = "data/processed",
    ) -> None:
        """Ingest ``raw_dir`` and build the BM25 index in ``index_dir``."""
        if isinstance(max_chunk_size, bool) or not isinstance(
            max_chunk_size, int
        ):
            raise ValueError("max_chunk_size must be an integer")
        if not 100 <= max_chunk_size <= 2000:
            raise ValueError("max_chunk_size must be between 100 and 2000")
        _check_dir(raw_dir, "raw_dir")
        _check_dir(index_dir, "index_dir")
        count = Utils().build_index(raw_dir, index_dir, max_chunk_size)
        print(
            f"Ingestion complete! Indexed {count} chunks under "
            f"{index_dir.rstrip('/')}/"
        )

    @_safe
    def search(
        self, query: str, k: int = 10, index_dir: str = "data/processed"
    ) -> None:
        """Print the top-k sources for a single query."""
        query = _check_query(query)
        k = _check_k(k)
        for c in Retriever(index_dir).search(query, k):
            print(
                f"{c.source} "
                f"[{c.first_character_index}:{c.last_character_index}]"
            )

    @_safe
    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        index_dir: str = "data/processed",
    ) -> None:
        """Search a whole dataset and write a StudentSearchResults JSON."""
        dataset_path = _check_json(dataset_path, "dataset_path")
        k = _check_k(k)
        save_directory = _check_dir(save_directory, "save_directory")
        with open(dataset_path, encoding="utf-8") as f:
            dataset = RagDataset.model_validate(json.load(f))
        retriever = Retriever(index_dir)

        results: List[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching", unit=" q"):
            hits = retriever.search(q.question, k)
            results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=[
                        MinimalSource(
                            file_path=c.source,
                            first_character_index=c.first_character_index,
                            last_character_index=c.last_character_index,
                        )
                        for c in hits
                    ],
                )
            )
        out = _save(
            StudentSearchResults(search_results=results, k=k),
            save_directory,
            os.path.basename(dataset_path),
        )
        print(f"Saved student_search_results to {out}")

    @_safe
    def answer(
        self, query: str, k: int = 5, index_dir: str = "data/processed"
    ) -> None:
        """Answer a single query using the retrieved context."""
        query = _check_query(query)
        k = _check_k(k)
        hits = Retriever(index_dir).search(query, k)
        if not hits:
            print("No relevant context found for this question.")
            return
        context = _build_context(
            [
                (
                    f"{c.source} [{c.first_character_index}:"
                    f"{c.last_character_index}]",
                    c.text,
                )
                for c in hits
            ]
        )
        from .llm import LLM

        print(LLM().ask_llm(question=query, context=context))

    @_safe
    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers from a StudentSearchResults JSON file."""
        path = _check_json(
            student_search_results_path, "student_search_results_path"
        )
        save_directory = _check_dir(save_directory, "save_directory")
        with open(path, encoding="utf-8") as f:
            student = StudentSearchResults.model_validate(json.load(f))

        from .llm import LLM

        llm = LLM()
        utils = Utils()
        cache: Dict[str, Optional[str]] = {}

        def source_text(src: MinimalSource) -> Optional[str]:
            """Read the text of a source from the corpus (cached)."""
            if src.file_path not in cache:
                try:
                    cache[src.file_path] = utils.read_text(src.file_path)
                except OSError:
                    cache[src.file_path] = None
            full = cache[src.file_path]
            if full is None:
                return None
            return full[
                src.first_character_index:src.last_character_index
            ]

        answers: List[MinimalAnswer] = []
        for res in tqdm(student.search_results, desc="Answering", unit=" q"):
            blocks: List[Tuple[str, str]] = []
            for src in res.retrieved_sources:
                text = source_text(src)
                if text:
                    blocks.append(
                        (
                            f"{src.file_path} [{src.first_character_index}:"
                            f"{src.last_character_index}]",
                            text,
                        )
                    )
            if blocks:
                reply = llm.ask_llm(
                    question=res.question, context=_build_context(blocks)
                )
            else:
                reply = "No relevant context found for this question."
            answers.append(
                MinimalAnswer(
                    question_id=res.question_id,
                    question=res.question,
                    retrieved_sources=res.retrieved_sources,
                    answer=reply,
                )
            )
        out = _save(
            StudentSearchResultsAndAnswer(
                search_results=answers, k=student.k
            ),
            save_directory,
            os.path.basename(path),
        )
        print(f"Saved student_search_results_and_answer to {out}")

    @_safe
    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Compute recall@k (IoU > 0.05) against a ground-truth dataset."""
        sp = _check_json(
            student_search_results_path, "student_search_results_path"
        )
        dp = _check_json(dataset_path, "dataset_path")
        with open(sp, encoding="utf-8") as f:
            student = StudentSearchResults.model_validate(json.load(f))
        with open(dp, encoding="utf-8") as f:
            dataset = RagDataset.model_validate(json.load(f))

        truth: Dict[str, List[MinimalSource]] = {
            q.question_id: q.sources
            for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion) and q.sources
        }
        cutoffs = sorted({c for c in (1, 3, 5, 10, student.k)
                          if c <= student.k})
        scores: Dict[int, List[float]] = {c: [] for c in cutoffs}
        for res in student.search_results:
            gt = truth.get(res.question_id)
            if not gt:
                continue
            for c in cutoffs:
                top = res.retrieved_sources[:c]
                found = sum(
                    1
                    for t in gt
                    if any(
                        r.file_path == t.file_path
                        and _iou(t, r) > IOU_THRESHOLD
                        for r in top
                    )
                )
                scores[c].append(found / len(gt))

        n = len(scores[cutoffs[0]])
        print(f"Questions evaluated: {n}")
        print(
            "  ".join(
                f"Recall@{c}: {(sum(v) / n if n else 0.0):.3f}"
                for c, v in scores.items()
            )
        )
