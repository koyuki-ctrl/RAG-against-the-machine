"""Corpus discovery, chunking, tokenization and BM25 retrieval."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bm25s
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from tqdm import tqdm

from .cache import DiskCache, index_version, make_key
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

MAX_CONTEXT_CHARS = 9000


def build_context(blocks: List[Tuple[str, str]]) -> str:
    """Format snippets for the prompt within a character budget.

    Args:
        blocks: ``(label, text)`` pairs, best first.

    Returns:
        The context string sent to the LLM.
    """
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


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON through a temp file so readers never see a partial file.

    Args:
        path: Destination file.
        data: JSON-serialisable content.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


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

    @staticmethod
    def file_hash(path: str) -> str:
        """Compute the SHA-256 of a file (used to detect changes).

        Args:
            path: File path.

        Returns:
            Hexadecimal digest.
        """
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def _hash_files(self, files: List[str]) -> Dict[str, str]:
        """Hash every file, skipping the unreadable ones.

        Args:
            files: File paths.

        Returns:
            Mapping ``path -> sha256``.
        """
        hashes: Dict[str, str] = {}
        for path in tqdm(files, desc="Hashing", unit=" file"):
            try:
                hashes[path] = self.file_hash(path)
            except OSError as exc:
                tqdm.write(f"[WARN] skipped {path}: {exc}")
        return hashes

    @staticmethod
    def _save_index(
        chunks: List[IndexedChunk],
        index_dir: str,
        max_chunk_size: int,
        hashes: Dict[str, str],
    ) -> None:
        """Tokenize, build the BM25 index and persist everything.

        Writes the BM25 files, ``chunks.json`` and ``manifest.json``.

        Args:
            chunks: All chunks of the corpus (in file order).
            index_dir: Output directory.
            max_chunk_size: Chunk size used to build the chunks.
            hashes: ``path -> sha256`` of every indexed file.
        """
        corpus_tokens = [
            tokenize(f"{c.source}\n{c.breadcrumb}\n{c.text}")
            for c in tqdm(chunks, desc="Tokenizing", unit=" chunk")
        ]
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, show_progress=False)

        out = Path(index_dir)
        out.mkdir(parents=True, exist_ok=True)
        bm25.save(str(out))
        _write_json_atomic(
            out / "chunks.json", [c.model_dump() for c in chunks]
        )
        # written last: a new manifest means a complete new index
        _write_json_atomic(
            out / "manifest.json",
            {"max_chunk_size": max_chunk_size, "files": hashes},
        )

    def build_index(
        self, raw_dir: str, index_dir: str, max_chunk_size: int
    ) -> int:
        """Chunk the whole corpus, build a BM25 index and persist it.

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
        hashes = self._hash_files(files)
        chunks = self.documents_chunkers(list(hashes), max_chunk_size)
        if not chunks:
            raise ValueError("The corpus produced no chunk")
        self._save_index(chunks, index_dir, max_chunk_size, hashes)
        return len(chunks)

    def update_index(
        self,
        raw_dir: str,
        index_dir: str,
        max_chunk_size: Optional[int] = None,
    ) -> Dict[str, int]:
        """Incrementally update an existing index.

        Only new and modified files are re-read and re-chunked. Chunks
        of unchanged files are reused as-is and chunks of deleted files
        are dropped. BM25 is then rebuilt from the chunks, which gives
        exactly the same index as a full rebuild.

        Args:
            raw_dir: Corpus directory.
            index_dir: Directory of the existing index.
            max_chunk_size: New chunk size; None keeps the current one
                (a different value re-chunks every file).

        Returns:
            Counters: added, modified, deleted, unchanged, chunks.

        Raises:
            FileNotFoundError: If no index exists yet.
        """
        out = Path(index_dir)
        manifest_file = out / "manifest.json"
        chunks_file = out / "chunks.json"
        if not manifest_file.is_file() or not chunks_file.is_file():
            raise FileNotFoundError(
                f"No index in {index_dir}: run the 'index' command first"
            )
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
        old_hashes: Dict[str, str] = manifest["files"]
        size = int(manifest["max_chunk_size"])
        rechunk_all = max_chunk_size is not None and max_chunk_size != size
        if max_chunk_size is not None:
            size = max_chunk_size

        files = self.list_valid_path(raw_dir, ".txt", ".md", ".py")
        new_hashes = self._hash_files(files)
        added = [f for f in new_hashes if f not in old_hashes]
        modified = [
            f for f in new_hashes
            if f in old_hashes
            and (rechunk_all or new_hashes[f] != old_hashes[f])
        ]
        deleted = [f for f in old_hashes if f not in new_hashes]

        with open(chunks_file, encoding="utf-8") as f:
            old_chunks = [
                IndexedChunk.model_validate(c) for c in json.load(f)
            ]
        stats = {
            "added": len(added),
            "modified": len(modified),
            "deleted": len(deleted),
            "unchanged": len(new_hashes) - len(added) - len(modified),
            "chunks": len(old_chunks),
        }
        if not (added or modified or deleted):
            return stats

        redo = set(added) | set(modified)
        by_source: Dict[str, List[IndexedChunk]] = {}
        for c in old_chunks:
            if c.source in new_hashes and c.source not in redo:
                by_source.setdefault(c.source, []).append(c)
        if redo:
            for c in self.documents_chunkers(sorted(redo), size):
                by_source.setdefault(c.source, []).append(c)
        chunks = [c for path in sorted(by_source) for c in by_source[path]]
        if not chunks:
            raise ValueError("The corpus produced no chunk")

        self._save_index(chunks, index_dir, size, new_hashes)
        stats["chunks"] = len(chunks)
        return stats


class Retriever:
    """BM25 retriever over the persisted index, with optional caching."""

    def __init__(
        self,
        index_dir: str = "data/processed",
        cache: Optional[DiskCache] = None,
        mmap: bool = True,
    ) -> None:
        """Load the index once.

        Args:
            index_dir: Directory written by ``Utils.build_index``.
            cache: Optional cache (fast chunk loading + query results).
            mmap: Memory-map the BM25 arrays (fast cold start). Use False
                in a long-running server that may rewrite the index.

        Raises:
            FileNotFoundError: If the index does not exist.
        """
        path = Path(index_dir)
        if not (path / "chunks.json").is_file():
            raise FileNotFoundError(
                f"No index in {index_dir}: run the 'index' command first"
            )
        self.cache = cache
        self.hits = 0
        self.misses = 0
        self._memory: Dict[str, List[int]] = {}
        self.version = index_version(path)
        self.chunks = self._load_chunks(path)
        try:
            self.bm25 = bm25s.BM25.load(str(path), mmap=mmap)
        except (OSError, ValueError):
            self.bm25 = bm25s.BM25.load(str(path))

    def _load_chunks(self, path: Path) -> List[IndexedChunk]:
        """Load chunks, from the fast cached copy when it is valid."""
        name = f"chunks-{self.version}"
        if self.cache is not None:
            stored = self.cache.load_pickle(name)
            if isinstance(stored, list):
                try:
                    return [IndexedChunk.model_construct(**d) for d in stored]
                except TypeError:
                    pass
        with open(path / "chunks.json", encoding="utf-8") as f:
            chunks = [IndexedChunk.model_validate(c) for c in json.load(f)]
        if self.cache is not None:
            self.cache.save_pickle(name, [c.model_dump() for c in chunks])
            self.cache.purge("chunks-", keep=name)
        return chunks

    def _search_ids(self, query: str, k: int) -> List[int]:
        """Run the real BM25 search and return chunk ids."""
        tokens = [t for t in tokenize(query) if t in self.bm25.vocab_dict]
        if not tokens:
            return []
        k = min(k, len(self.chunks))
        results, _ = self.bm25.retrieve([tokens], k=k, show_progress=False)
        return [int(i) for i in results[0]]

    def search(self, query: str, k: int) -> List[IndexedChunk]:
        """Return the top-k chunks for a query (cached when possible).

        The cache key contains the normalised query, ``k`` and the index
        version, so an ``index`` or ``update`` invalidates old results.

        Args:
            query: Natural-language question.
            k: Number of chunks wanted.

        Returns:
            Chunks ranked by decreasing BM25 score (may be empty).
        """
        ids: Optional[List[int]] = None
        key = make_key(self.version, str(k), " ".join(query.lower().split()))
        if self.cache is not None:
            ids = self._memory.get(key)
            if ids is None:
                stored = self.cache.get(key)
                if isinstance(stored, list) and all(
                    isinstance(i, int) and 0 <= i < len(self.chunks)
                    for i in stored
                ):
                    ids = stored
        if ids is not None:
            self.hits += 1
        else:
            self.misses += 1
            ids = self._search_ids(query, k)
            if self.cache is not None:
                self.cache.set(key, ids)
        if self.cache is not None:
            self._memory[key] = ids
        return [self.chunks[i] for i in ids]
