from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import orjson
import tiktoken

from .documents import Document


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
    def chunk_text(self, piece: Union[Document, Chunk]) -> List[Chunk]:
        raise NotImplementedError()

    def generate_chunks(self, pieces: Iterator[Union[Document, Chunk]]):
        for piece in pieces:
            yield from self.chunk_text(piece)

    def __call__(self, pieces: Iterator[Union[Document, Chunk]]):
        return self.generate_chunks(pieces)


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

    def chunk_text(self, piece: Union[Document, Chunk]):
        text = piece.read()
        if isinstance(piece, Document):
            doc_id = piece.id
            metadata = piece.metadata
        else:
            doc_id = piece.doc_id
            metadata = piece.metadata

        chunks: List[Chunk] = []
        token_ids = self.tokenizer.encode(text, disallowed_special=())

        start_char_idx = metadata.get("start_char_idx", 0)
        for start in range(0, len(token_ids), self.max_tokens - self.overlap):
            end = start + self.max_tokens
            chunk_text = self.tokenizer.decode(token_ids[start:end])

            chunks.append(
                Chunk(
                    doc_id,
                    chunk_text,
                    embedding=None,
                    metadata={
                        **metadata,
                        "start_char_idx": start_char_idx,
                        "end_char_idx": start_char_idx + len(chunk_text),
                    },
                )
            )

            non_overlapping = self.tokenizer.decode(
                token_ids[start : start + self.max_tokens - self.overlap]
            )
            start_char_idx += len(non_overlapping)

        return chunks


if __name__ == "__main__":
    text = """
This is the text to perform chunking on.

Firstly based on the number of tokens.
"""

    chunker = FixedTokenChunker("cl100k_base", max_tokens=5, overlap=0)
    chunks = chunker.chunk_text(Chunk(doc_id=1, text=text))
    print(chunks)
    print([len(chunk.text) for chunk in chunks])

    merged = Chunk.merge_chunks(
        Chunk(doc_id=0, text="This is a chunk", metadata={"start_char_idx": 0, "end_char_idx": 15}),
        Chunk(doc_id=0, text="is a chunk continuation", metadata={"start_char_idx": 5, "end_char_idx": 28}),
    )
    print(merged)
