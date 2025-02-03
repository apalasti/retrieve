from abc import ABC, abstractmethod
from typing import Iterator, List, Dict

import numpy as np

from retrieve.core.chunking import Chunk


def make_batches(iterator: Iterator, batch_size: int):
    batch = []
    for item in iterator:
        if len(batch) < batch_size:
            batch.append(item)
        else:
            yield batch
            batch = [item]
    if batch:
        yield batch


class Embedder(ABC):
    EMBEDDING_BATCH_SIZE = 1

    @abstractmethod
    def get_embedding_dims(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def tokenize(self, text: str) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def embed_texts(
        self,
        texts: List[str],
        output_value="sentence_embedding",
        prompt_name=None,
    ) -> List[np.ndarray]:
        raise NotImplementedError()

    def __call__(
        self,
        chunks: Iterator[Chunk],
        output_value="sentence_embedding",
    ):
        for batch in make_batches(chunks, self.EMBEDDING_BATCH_SIZE):
            texts = [chunk.text for chunk in batch]
            embeddings = self.embed_texts(texts, output_value)
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
            yield batch


class HFEmbedding(Embedder):
    BATCH_SIZE = 1

    def __init__(self, model_name: str, **kwargs) -> None:
        # import torch
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, **kwargs)
        # self._model = torch.compile(self._model, backend="tvm", mode="max-autotune")

    def tokenize(self, text: str, prompt_name=None) -> Dict:
        if prompt_name is None:
            prompt_name = self._model.default_prompt_name
        if prompt_name is not None:
            text = self._model.prompts[prompt_name] + text

        return self._model.tokenizer(
            text,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            return_attention_mask=False,
        )

    def get_embedding_dims(self):
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(
        self, texts: List[str], output_value="sentence_embedding", prompt_name=None
    ) -> List[np.ndarray]:
        prompt_name = prompt_name if prompt_name in self._model.prompts else None
        return list(
            self._model.encode(
                sentences=texts,
                batch_size=self.BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                output_value=output_value,
                prompt_name=prompt_name
            )
        )


if __name__ == "__main__":
    embedder = HFEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"torch_dtype": "float16"},
    )
    print(
        [
            v.shape
            for v in embedder.embed_texts(
                ["this is a sentence i want to embed", "this is another one"]
            )
        ]
    )
