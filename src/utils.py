from pathlib import Path
from typing import List, Any
import errno
import os
import bm25s
import json
from tqdm import tqdm
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
)
from langchain_community.document_loaders import (
    TextLoader,
    PythonLoader,
)
from langchain_core.documents import Document


class Utils:
    def list_valid_path(self, path: str, *args) -> List[str]:
        fls = []
        exist_directory = Path(path).is_dir()
        if exist_directory:
            directory = Path(path)
            for paths, _, files in os.walk(directory):
                for file in files:
                    for arg in args:
                        if file.endswith(str(arg)):
                            fls.append(os.path.join(paths, file))
            return fls
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path
            )

    def txt_splitter(
        self,
        file_path: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> List[Document]:
        docs = TextLoader(file_path, encoding="utf-8").load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)
        return chunks

    def markdown_splitter(
        self,
        file_path: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> List[Document]:
        docs = TextLoader(file_path, encoding="utf-8").load()
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"), ("##", "h2"), ("###", "h3"),
            ]
        )
        md_docs = md_splitter.split_text(docs[0].page_content)
        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )
        chunks = size_splitter.split_documents(md_docs)
        return chunks

    def python_splitter(
        self,
        file_path: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> List[Document]:
        docs = PythonLoader(file_path).load()
        splitter = PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)
        return chunks

    def documents_chunkers(
        self, files, max_chunk_size
    ) -> List[Document]:
        all_chunks: List[Document] = []
        for file_path in tqdm(files, desc="Chunking: ", unit=" file"):
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".txt":
                chunks = self.txt_splitter(file_path, max_chunk_size)
            elif ext == ".md":
                chunks = self.markdown_splitter(file_path, max_chunk_size)
            elif ext == ".py":
                chunks = self.python_splitter(file_path, max_chunk_size)
            else:
                continue

            for c in chunks:
                c.metadata["extension"] = ext
                c.metadata["source"] = file_path
            all_chunks.extend(chunks)

        return all_chunks

    def load_retriever(
        self, index_path: str = "data/processed/"
    ) -> tuple[Any, list[dict[str, Any]]]:
        retriever = bm25s.BM25.load(index_path, load_corpus=True)

        # Fallback : si bm25s n'a pas rechargé le corpus, on le lit à la main
        if retriever.corpus is None:
            corpus_path = Path(index_path) / "corpus.json"
            with open(corpus_path, encoding="utf-8") as f:
                retriever.corpus = json.load(f)

        with open(f"{index_path}/metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        return retriever, metadata

    def retrieve_result(
        self, retriever: Any, query: str, k: int
    ) -> tuple[Any, Any]:
        query_token = bm25s.tokenize(query)
        results, scores = retriever.retrieve(query_token, k=k)
        return results, scores

    def extract_hit(
        self,
        retriever: Any,
        metadata: list[dict[str, Any]],
        item: Any,
    ) -> tuple[int, dict[str, Any], str]:
        """Retourne (doc_id, meta, texte) pour un hit bm25s.

        Gère les deux formats d'API :
        - ancien : item est un int (index dans corpus)
        - nouveau : item est un dict avec 'id' et éventuellement 'text'
        """
        if isinstance(item, dict):
            doc_id = int(item.get("id", item.get("doc_id")))
            chunk_text = item.get("text")
            if chunk_text is None and retriever.corpus is not None:
                chunk_text = retriever.corpus[doc_id]
        else:
            doc_id = int(item)
            chunk_text = (
                retriever.corpus[doc_id]
                if retriever.corpus is not None
                else ""
            )

        meta = metadata[doc_id]
        return doc_id, meta, chunk_text
