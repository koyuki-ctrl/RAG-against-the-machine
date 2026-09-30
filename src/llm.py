from llama_cpp import Llama
import re


class LLM:
    def __init__(
        self,
        repo_id: str = "Qwen/Qwen3-0.6B-GGUF",
        filename: str = "*Q8_0.gguf",
        n_ctx: int = 8192,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self.repo_id = repo_id
        self.filename = filename
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            temperature=temperature,
            verbose=False,
        )

    @staticmethod
    def _strip_thinking(raw: str) -> str:
        """Retire tous les blocs de raisonnement (balises + contenu).

        Gère les variantes :
        - <think> ... </think>
        - <|think|> ... <|/think|>  (selon le tokenizer)
        - bloc non fermé (max_tokens coupe la génération)
        """
        clean = raw

        # 1. Blocs complets
        clean = re.sub(
            r"<think>.*?</think>", "", clean, flags=re.DOTALL
        )
        clean = re.sub(
            r"<\|think\|>.*?<\|/think\|>", "", clean, flags=re.DOTALL
        )

        # 2. Blocs non fermés (ouverture sans fermeture)
        clean = re.sub(r"<think>.*$", "", clean, flags=re.DOTALL)
        clean = re.sub(r"<\|think\|>.*$", "", clean, flags=re.DOTALL)

        # 3. Tokens spéciaux résiduels
        clean = re.sub(r"<\|im_end\|>", "", clean)
        clean = re.sub(r"<\|im_start\|>", "", clean)
        clean = re.sub(r"<\|endoftext\|>", "", clean)

        return clean.strip()

    def ask_llm(self, question: str, context: str) -> str:
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using ONLY the context. "
                        f"Context:\n{context}"
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        raw = str(response["choices"][0]["message"]["content"])
        return self._strip_thinking(raw)
