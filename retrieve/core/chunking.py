from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import orjson
import tiktoken

from retrieve.core.documents import Document


@dataclass
class Chunk:
    doc_id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def read(self):
        return self.text

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": orjson.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(
            doc_id=d["doc_id"],
            text=d["text"],
            embedding=d["embedding"],
            metadata=orjson.loads(d["metadata"]),
        )

    @staticmethod
    def merge_chunks(*chunks: "Chunk") -> List["Chunk"]:

        def clone_chunk(chunk: Chunk):
            return Chunk(
                doc_id=chunk.doc_id,
                text=chunk.text,
                embedding=chunk.embedding,
                metadata=dict(**chunk.metadata),
            )

        chunks = sorted(
            chunks, key=lambda chunk: (chunk.doc_id, chunk.metadata["start_char_idx"])
        )
        merged = [clone_chunk(chunks[0])]
        for chunk in chunks:
            overlaps = (
                merged[-1].metadata["end_char_idx"] - chunk.metadata["start_char_idx"]
            )
            if merged[-1].doc_id == chunk.doc_id and 0 <= overlaps:
                merged[-1].text += chunk.text[overlaps:]
                merged[-1].metadata["end_char_idx"] += len(chunk.text) - overlaps
                merged[-1].metadata["score"] = merged[-1].metadata.get(
                    "score", 0
                ) + chunk.metadata.get("score", 0)
            else:
                merged.append(clone_chunk(chunk))
        return merged

    def __repr__(self):
        text = self.text[:20] + "..." if 20 < len(self.text) else self.text
        return f"Chunk(doc_id='{self.doc_id}', text='{text}', metadata={self.metadata})"


class Chunker(ABC):
    @abstractmethod
    def get_chunk_boundaries(self, text: str) -> List[Tuple[int, int]]:
        raise NotImplementedError()

    def chunk_node(self, node: Union[Document, Chunk]) -> List[Chunk]:
        text = node.read()
        if isinstance(node, Document):
            doc_id = node.id
            metadata = node.metadata
        else:
            doc_id = node.doc_id
            metadata = node.metadata

        start_char_idx = metadata.get("start_char_idx", 0)
        return [
            Chunk(
                doc_id,
                text[start:end],
                embedding=None,
                metadata={
                    **metadata,
                    "start_char_idx": start_char_idx + start,
                    "end_char_idx": start_char_idx + end,
                },
            )
            for (start, end) in self.get_chunk_boundaries(text)
        ]

    def __call__(self, nodes: Iterator[Union[Document, Chunk]]):
        for node in nodes:
            yield from self.chunk_node(node)


class FixedTokenChunker(Chunker):
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        max_tokens: int = 250,
        overlap: int = 0,
    ) -> None:
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.max_tokens = max_tokens
        self.overlap = overlap

    def get_chunk_boundaries(self, text: str) -> List[Tuple[int, int]]:
        tokens = self.tokenizer.encode(text, disallowed_special=())

        batches = []
        for start in range(0, len(tokens), self.max_tokens - self.overlap):
            non_overlapping = tokens[start : start + self.max_tokens - self.overlap]
            overlapping = tokens[
                start + self.max_tokens - self.overlap : start + self.max_tokens
            ]
            batches.append(non_overlapping)
            batches.append(overlapping)

        parts = self.tokenizer.decode_batch(batches)
        bounds, start_char_idx = [], 0
        for non_overlapping, overlapping in zip(parts[::2], parts[1::2]):
            end_char_idx = start_char_idx + len(non_overlapping) + len(overlapping)
            bounds.append((start_char_idx, end_char_idx))
            start_char_idx += len(non_overlapping)
        return bounds


if __name__ == "__main__":
    text = """
This is the text to perform chunking on.

Firstly based on the number of tokens.
"""

    chunker = FixedTokenChunker("cl100k_base", max_tokens=5, overlap=1)
    chunks = chunker.chunk_node(Chunk(doc_id=1, text=text))
    print(chunks)
    print([len(chunk.text) for chunk in chunks])
    print("Bounds: ", chunker.get_chunk_boundaries(text))

    merged = Chunk.merge_chunks(*chunks)
    print(merged)
    print(merged[0].text)
