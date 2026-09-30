"""Answer generation with Qwen3-0.6B through llama.cpp."""
import re
from typing import Any, Optional

from .cache import DiskCache, make_key

PROMPT_VERSION = "1"


class LLM:
    """Thin wrapper around a local GGUF model (lazy loading + cache)."""

    def __init__(
        self,
        repo_id: str = "Qwen/Qwen3-0.6B-GGUF",
        filename: str = "*Q8_0.gguf",
        n_ctx: int = 8192,
        temperature: float = 0.1,
        max_tokens: int = 512,
        cache: Optional[DiskCache] = None,
    ) -> None:
        """Store the settings; the model is loaded on first real use.

        Args:
            repo_id: Hugging Face repository of the GGUF model.
            filename: Glob selecting the GGUF file.
            n_ctx: Context window size in tokens.
            temperature: Sampling temperature.
            max_tokens: Maximum number of generated tokens.
            cache: Optional cache of generated answers.
        """
        self.repo_id = repo_id
        self.filename = filename
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.hits = 0
        self.misses = 0
        self._model: Any = None

    def _load(self) -> Any:
        """Load the model once (never when every answer is cached)."""
        if self._model is None:
            from llama_cpp import Llama

            self._model = Llama.from_pretrained(
                repo_id=self.repo_id,
                filename=self.filename,
                n_ctx=self.n_ctx,
                verbose=False,
            )
        return self._model

    @staticmethod
    def _strip_thinking(raw: str) -> str:
        """Remove reasoning blocks and leftover special tokens.

        Args:
            raw: Raw model output.

        Returns:
            The cleaned answer.
        """
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        clean = re.sub(r"<think>.*$", "", clean, flags=re.DOTALL)
        clean = re.sub(r"<\|(im_end|im_start|endoftext)\|>", "", clean)
        return clean.strip()

    def _generate(self, question: str, context: str) -> str:
        """Call the model."""
        system = (
            "You answer questions about the vLLM codebase. "
            "Answer using ONLY the context below. If the context does "
            "not contain the answer, say you cannot find it. "
            "Be concise and precise.\n\n"
            f"Context:\n{context}"
        )
        response = self._load().create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{question} /no_think"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw = str(response["choices"][0]["message"]["content"])
        return self._strip_thinking(raw)

    def ask_llm(self, question: str, context: str) -> str:
        """Answer a question from the retrieved context only.

        The cache key contains the model, its parameters, the prompt
        version, the question and the exact context.

        Args:
            question: The user question.
            context: Retrieved snippets, already formatted.

        Returns:
            The generated answer.
        """
        key = make_key(
            "llm", PROMPT_VERSION, self.repo_id, self.filename,
            str(self.n_ctx), str(self.temperature), str(self.max_tokens),
            question, context,
        )
        if self.cache is not None:
            cached = self.cache.get(key)
            if isinstance(cached, str):
                self.hits += 1
                return cached
        self.misses += 1
        answer = self._generate(question, context)
        if self.cache is not None and answer:
            self.cache.set(key, answer)
        return answer
