from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np

from src.chunking import Chunk


def make_batches(iterator: Iterator, batch_size: int):
    batch = []
    for item in iterator:
        if len(batch) < batch_size:
            batch.append(item)
        else:
            yield batch
            batch = []
    if batch:
        yield batch


class Embedder(ABC):
    @abstractmethod
    def get_embedding_dims(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        raise NotImplementedError()

    def __call__(self, chunks: Iterator[Chunk]):
        for batch in make_batches(chunks, 100):
            texts = [chunk.text for chunk in batch]
            embeddings = self.embed_texts(texts)
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
            yield batch


class STEmbedding(Embedder):
    BATCH_SIZE = 50

    def __init__(self, model_name: str, **kwargs) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, **kwargs)
        self.model = torch.compile(self.model, backend="tvm", mode="max-autotune")

    def get_embedding_dims(self):
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        return list(
            self.model.encode(
                sentences=texts,
                batch_size=self.BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        )


if __name__ == "__main__":
    embedder = STEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"torch_dtype": "float16"}
    )
    print(
        [
            v.shape
            for v in embedder.embed_texts(
                ["this is a sentence i want to embed", "this is another one"]
            )
        ]
    )
