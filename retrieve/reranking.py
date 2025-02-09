from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np

from retrieve.core import Chunk


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(self, query_text: str, chunks: List[Chunk]):
        """
        Reranks a list of chunks based on the given query, by setting the score of each chunk based on relevancy.
        """
        pass


class LBReranker(Reranker):
    def __init__(self, precision="Q4_0", n_ctx=512, **kwargs) -> None:
        from llama_cpp import Llama
        kwargs.setdefault("verbose", False)
        self._model = Llama.from_pretrained(
            repo_id="bartowski/lb-reranker-0.5B-v1.0-GGUF",
            filename=f"*{precision}.gguf",
            logits_all=True,
            n_ctx=n_ctx,
            **kwargs,
        )

    def create_chat(self, query: str, text: str):
        system_message = "Given a query and a piece of text, output a score of 1-7 based on how related the query is to the text. 1 means least related and 7 is most related."
        user_message = "<<<Query>>>\n{query}\n\n<<<Context>>>\n{text}"
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message.format(query=query, text=text)},
        ]

    def calculate_score(self, logprobs_dict: Dict[str, np.float32]):
        scores = [
            float(token) * np.exp(logprob)
            for token, logprob in logprobs_dict.items()
            if token.isdecimal()
        ]
        return float(np.sum(scores)) if scores else 0.0

    def rerank(self, query_text: str, chunks: List[Chunk]):
        for chunk in chunks:
            result = self._model.create_chat_completion(
                self.create_chat(query_text, chunk.text),
                temperature=0.0,
                logprobs=True,
                top_logprobs=10,
                top_p=0,
            )
            top_logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
            chunk.metadata["score"] = self.calculate_score(top_logprobs)

