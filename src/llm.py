"""Answer generation with Qwen3-0.6B through llama.cpp."""
import re
from typing import Any


class LLM:
    """Thin wrapper around a local GGUF model."""

    def __init__(
        self,
        repo_id: str = "Qwen/Qwen3-0.6B-GGUF",
        filename: str = "*Q8_0.gguf",
        n_ctx: int = 8192,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        """Load the model (downloaded from the HF hub on first use).

        Args:
            repo_id: Hugging Face repository of the GGUF model.
            filename: Glob selecting the GGUF file.
            n_ctx: Context window size in tokens.
            temperature: Sampling temperature.
            max_tokens: Maximum number of generated tokens.
        """
        from llama_cpp import Llama

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm: Any = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            verbose=False,
        )

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

    def ask_llm(self, question: str, context: str) -> str:
        """Answer a question from the retrieved context only.

        Args:
            question: The user question.
            context: Retrieved snippets, already formatted.

        Returns:
            The generated answer.
        """
        system = (
            "You answer questions about the vLLM codebase. "
            "Answer using ONLY the context below. If the context does "
            "not contain the answer, say you cannot find it. "
            "Be concise and precise.\n\n"
            f"Context:\n{context}"
        )
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{question} /no_think"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw = str(response["choices"][0]["message"]["content"])
        return self._strip_thinking(raw)
