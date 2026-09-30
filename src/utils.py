"""Corpus discovery, chunking, tokenization and BM25 retrieval."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import bm25s
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from tqdm import tqdm

from .models import IndexedChunk

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i if in into is "
    "it its of on or that the their then there these this to was what "
    "when where which who why will with you your do does can should "
    "would could about used use using".split()
)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")


def tokenize(text: str) -> List[str]:
    """Tokenize text and code for BM25.

    Identifiers are kept whole (``max_model_len``) AND split into their
    parts (``max``, ``model``, ``len``), so both a verbatim identifier
    and a paraphrase in natural language can match.

    Args:
        text: Text or source code.

    Returns:
        Lower-case tokens without stopwords.
    """
    tokens: List[str] = []
    for match in _IDENT_RE.finditer(text):
        word = match.group(0).strip("_")
        if len(word) < 2:
            continue
        pieces: List[str] = []
        for part in word.split("_"):
            pieces.extend(_CAMEL_RE.split(part))
        pieces = [p.lower() for p in pieces if len(p) >= 2]
        tokens.append(word.lower())
        if len(pieces) > 1:
            tokens.extend(pieces)
    return [t for t in tokens if t not in STOPWORDS]


class Utils:
    """File discovery and chunking helpers."""

    @staticmethod
    def read_text(path: str) -> str:
        """Read a file the same way at index time and at answer time.

        Args:
            path: File path.

        Returns:
            The decoded text.
        """
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def list_valid_path(self, path: str, *extensions: str) -> List[str]:
        """List files under ``path`` with one of the given extensions.

        Args:
            path: Root directory.
            *extensions: Accepted suffixes (e.g. ``".md"``).

        Returns:
            Sorted list of file paths (prefixed by ``path``).

        Raises:
            FileNotFoundError: If ``path`` is not a directory.
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")
        files: List[str] = []
        for folder, _, names in os.walk(os.path.normpath(path)):
            for name in names:
                if name.endswith(extensions):
                    files.append(os.path.join(folder, name))
        return sorted(files)

    @staticmethod
    def _split(text: str, ext: str, chunk_size: int) -> List[Document]:
        """Split a text with the strategy matching its extension.

        Args:
            text: Full file content.
            ext: File extension (``.py``, ``.md``, ``.txt``).
            chunk_size: Maximum chunk size in characters.

        Returns:
            Documents carrying ``start_index`` in their metadata.
        """
        overlap = min(200, chunk_size // 5)
        if ext == ".py":
            splitter = RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                add_start_index=True,
            )
        elif ext == ".md":
            splitter = RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                add_start_index=True,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                add_start_index=True,
            )
        return splitter.create_documents([text])

    @staticmethod
    def _markdown_headers(text: str) -> List[Tuple[int, int, str]]:
        """Find markdown headers (ignoring fenced code blocks).

        Args:
            text: Markdown content.

        Returns:
            List of ``(char_position, level, title)``.
        """
        headers: List[Tuple[int, int, str]] = []
        offset = 0
        in_fence = False
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
            elif not in_fence:
                m = _HEADER_RE.match(line.rstrip("\r\n"))
                if m:
                    headers.append((offset, len(m.group(1)), m.group(2)))
            offset += len(line)
        return headers

    @staticmethod
    def _breadcrumb(
        headers: List[Tuple[int, int, str]], start: int
    ) -> str:
        """Build the ``h1 > h2 > h3`` path active at a position.

        Args:
            headers: Output of ``_markdown_headers``.
            start: Character position of the chunk.

        Returns:
            The header path, or an empty string.
        """
        stack: dict[int, str] = {}
        for pos, level, title in headers:
            if pos > start:
                break
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = title
        return " > ".join(stack[k] for k in sorted(stack))

    def documents_chunkers(
        self, files: List[str], max_chunk_size: int
    ) -> List[IndexedChunk]:
        """Chunk every file.

        ``first/last_character_index`` always refer to the ORIGINAL file
        content and ``text`` is never modified, so the reported span is
        exact and never longer than ``max_chunk_size``.

        Args:
            files: Files to chunk.
            max_chunk_size: Maximum chunk size in characters.

        Returns:
            All chunks.
        """
        chunks: List[IndexedChunk] = []
        for file_path in tqdm(files, desc="Chunking", unit=" file"):
            try:
                text = self.read_text(file_path)
            except OSError as exc:
                tqdm.write(f"[WARN] skipped {file_path}: {exc}")
                continue
            if not text.strip():
                continue
            ext = os.path.splitext(file_path)[1].lower()
            headers = self._markdown_headers(text) if ext == ".md" else []
            for doc in self._split(text, ext, max_chunk_size):
                content = doc.page_content[:max_chunk_size]
                if not content.strip():
                    continue
                start = max(0, int(doc.metadata.get("start_index", 0)))
                chunks.append(
                    IndexedChunk(
                        source=file_path,
                        first_character_index=start,
                        last_character_index=start + len(content),
                        extension=ext,
                        breadcrumb=self._breadcrumb(headers, start),
                        text=content,
                    )
                )
        return chunks

    def build_index(
        self, raw_dir: str, index_dir: str, max_chunk_size: int
    ) -> int:
        """Chunk the corpus, build a BM25 index and persist it.

        Args:
            raw_dir: Corpus directory.
            index_dir: Where the index is written.
            max_chunk_size: Maximum chunk size in characters.

        Returns:
            Number of indexed chunks.
        """
        files = self.list_valid_path(raw_dir, ".txt", ".md", ".py")
        if not files:
            raise FileNotFoundError(f"No .txt/.md/.py file in {raw_dir}")
        chunks = self.documents_chunkers(files, max_chunk_size)
        if not chunks:
            raise ValueError("The corpus produced no chunk")

        corpus_tokens = [
            tokenize(f"{c.source}\n{c.breadcrumb}\n{c.text}")
            for c in tqdm(chunks, desc="Tokenizing", unit=" chunk")
        ]
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, show_progress=False)

        out = Path(index_dir)
        out.mkdir(parents=True, exist_ok=True)
        bm25.save(str(out))
        with open(out / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f,
                      ensure_ascii=False)
        return len(chunks)


class Retriever:
    """BM25 retriever over the persisted index."""

    def __init__(self, index_dir: str = "data/processed") -> None:
        """Load the index once.

        Args:
            index_dir: Directory written by ``Utils.build_index``.

        Raises:
            FileNotFoundError: If the index does not exist.
        """
        path = Path(index_dir)
        chunks_file = path / "chunks.json"
        if not chunks_file.is_file():
            raise FileNotFoundError(
                f"No index in {index_dir}: run the 'index' command first"
            )
        with open(chunks_file, encoding="utf-8") as f:
            self.chunks = [
                IndexedChunk.model_validate(c) for c in json.load(f)
            ]
        self.bm25 = bm25s.BM25.load(str(path))

    def search(self, query: str, k: int) -> List[IndexedChunk]:
        """Return the top-k chunks for a query.

        Args:
            query: Natural-language question.
            k: Number of chunks wanted.

        Returns:
            Chunks ranked by decreasing BM25 score (may be empty).
        """
        tokens = [t for t in tokenize(query) if t in self.bm25.vocab_dict]
        if not tokens:
            return []
        k = min(k, len(self.chunks))
        results, _ = self.bm25.retrieve(
            [tokens], k=k, show_progress=False
        )
        return [self.chunks[int(i)] for i in results[0]]
