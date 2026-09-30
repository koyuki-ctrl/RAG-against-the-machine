from __future__ import annotations
from .utils import Utils
from .models import (
    RagDataset,
    MinimalSearchResults,
    MinimalSource,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .llm import LLM
from pathlib import Path
import bm25s
import json
import os
from tqdm import tqdm
from .models import AnsweredQuestion


class Arguments:
    @staticmethod
    def _overlap(a: MinimalSource, b: MinimalSource) -> bool:
        return not (
            a.last_character_index <= b.first_character_index
            or b.last_character_index <= a.first_character_index
        )

    def index(self, max_chunk_size: int) -> None:
        if not isinstance(max_chunk_size, int):
            raise ValueError("Error: your max_chunk must be a number")
        if max_chunk_size < 200:
            raise ValueError(
                f"Got a larger chunk overlap (200) than chunk size "
                f"{max_chunk_size}, should be smaller."
            )

        path = "data/raw/"
        files = Utils().list_valid_path(path, ".txt", ".md", ".py")
        chunks = Utils().documents_chunkers(files, max_chunk_size)
        corpus = [chunk.page_content for chunk in chunks]

        metadata = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            start = chunk.metadata.get("start_index", 0)
            end = start + len(chunk.page_content)
            metadata.append({
                "source": source,
                "first_character_index": start,
                "last_character_index": end,
                "extension": chunk.metadata.get("extension", ""),
            })

        tokens_corpus = bm25s.tokenize(
            corpus, lower=True, show_progress=True, stopwords="en"
        )
        retriever = bm25s.BM25()
        retriever.index(tokens_corpus)

        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        retriever.save(str(out_dir), corpus=corpus)

        (out_dir / "corpus.json").write_text(
            json.dumps(corpus, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int) -> None:
        if not isinstance(query, str) or not isinstance(k, int):
            raise ValueError("[ERROR]You must include the valid type")
        if not query:
            raise ValueError("[ERROR]No query is added")
        if not k:
            raise ValueError("[ERROR]No top-k is added")

        utils = Utils()
        retriever, metadata = utils.load_retriever("data/processed/")
        results, _ = utils.retrieve_result(retriever, query, k)

        for item in results[0]:
            _, meta, _ = utils.extract_hit(retriever, metadata, item)
            location = (
                f"{meta['source']} "
                f"[{meta['first_character_index']}:"
                f"{meta['last_character_index']}]"
            )
            print(location)

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        if not all(isinstance(x, t) for x, t in [
            (dataset_path, str), (k, int), (save_directory, str)
        ]):
            raise ValueError("[ERROR]You must include the valid type")
        if not dataset_path:
            raise ValueError("[ERROR]No dataset_path is added")
        if not k:
            raise ValueError("[ERROR]No top-k is added")
        if not save_directory:
            raise ValueError("[ERROR]No save_directory is added")
        if not dataset_path.endswith(".json"):
            raise ValueError("[ERROR]Only json files are authorized")

        basename = os.path.basename(dataset_path)
        with open(dataset_path, encoding="utf-8") as f:
            json_loader = json.load(f)
        validate_data = RagDataset.model_validate(json_loader)

        utils = Utils()
        retriever, metadata = utils.load_retriever("data/processed/")

        search_results: list[MinimalSearchResults] = []

        for question in validate_data.rag_questions:
            results, _ = utils.retrieve_result(
                retriever, question.question, k
            )

            retrieved_sources = []
            for item in results[0]:
                _, meta, _ = utils.extract_hit(retriever, metadata, item)
                retrieved_sources.append(
                    MinimalSource(
                        file_path=meta["source"],
                        first_character_index=meta["first_character_index"],
                        last_character_index=meta["last_character_index"],
                    )
                )

            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=retrieved_sources,
                )
            )

        output = StudentSearchResults(search_results=search_results, k=k)
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / basename
        out_path.write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Saved student_search_results to {out_path}")

    def answer(self, query: str, k: int = 5) -> None:
        if not isinstance(query, str) or not isinstance(k, int):
            raise ValueError("[ERROR]You must include the valid type")
        if not query:
            raise ValueError("[ERROR]No question is added")
        if not k:
            raise ValueError("[ERROR]No top-k is added")

        utils = Utils()
        retriever, metadata = utils.load_retriever("data/processed/")
        results, _ = utils.retrieve_result(retriever, query, k)

        context_parts = []
        for item in tqdm(results[0], desc="retrieve", unit=" file"):
            _, meta, chunk_text = utils.extract_hit(
                retriever, metadata, item
            )
            context_parts.append(
                f"Source: {meta['source']} "
                f"[{meta['first_character_index']}:"
                f"{meta['last_character_index']}]\n"
                f"{chunk_text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        answer = LLM().ask_llm(question=query, context=context)
        print(answer)

    def answer_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        if not all(isinstance(x, t) for x, t in [
            (dataset_path, str), (k, int), (save_directory, str)
        ]):
            raise ValueError("[ERROR]You must include the valid type")
        if not dataset_path:
            raise ValueError("[ERROR]No dataset_path is added")
        if not k:
            raise ValueError("[ERROR]No top-k is added")
        if not save_directory:
            raise ValueError("[ERROR]No save_directory is added")
        if not dataset_path.endswith(".json"):
            raise ValueError("[ERROR]Only json files are authorized")

        basename = os.path.basename(dataset_path)
        with open(dataset_path, encoding="utf-8") as f:
            json_loader = json.load(f)
        validate_data = RagDataset.model_validate(json_loader)

        utils = Utils()
        retriever, metadata = utils.load_retriever("data/processed/")
        llm = LLM()

        answers: list[MinimalAnswer] = []

        for question in tqdm(
            validate_data.rag_questions, desc="answer", unit=" q"
        ):
            results, _ = utils.retrieve_result(
                retriever, question.question, k
            )

            retrieved_sources = []
            context_parts = []
            for item in results[0]:
                _, meta, chunk_text = utils.extract_hit(
                    retriever, metadata, item
                )
                retrieved_sources.append(
                    MinimalSource(
                        file_path=meta["source"],
                        first_character_index=meta["first_character_index"],
                        last_character_index=meta["last_character_index"],
                    )
                )
                context_parts.append(chunk_text)

            context = "\n\n---\n\n".join(context_parts)
            answer = llm.ask_llm(
                question=question.question, context=context
            )

            answers.append(
                MinimalAnswer(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=retrieved_sources,
                    answer=answer,
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=k
        )
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / basename
        out_path.write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Saved student_answers to {out_path}")

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        if not (isinstance(student_search_results_path, str)
                and isinstance(dataset_path, str)):
            raise ValueError("[ERROR]You must include the valid type")

        with open(student_search_results_path, encoding="utf-8") as f:
            student = StudentSearchResults.model_validate(json.load(f))
        with open(dataset_path, encoding="utf-8") as f:
            dataset = RagDataset.model_validate(json.load(f))

        gt: dict[str, list[MinimalSource]] = {
            q.question_id: q.sources
            for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion) and q.sources
        }

        recalls: list[float] = []
        for res in student.search_results:
            truth = gt.get(res.question_id)
            if not truth:
                continue

            hits = 0
            for t in truth:
                if any(
                    r.file_path == t.file_path and self._overlap(t, r)
                    for r in res.retrieved_sources
                ):
                    hits += 1

            recalls.append(hits / len(truth))

        avg = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"Recall@{student.k} = {avg:.4f}  ({len(recalls)} questions)")
        return avg
