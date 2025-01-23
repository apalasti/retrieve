from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Iterator

import orjson
import numpy as np
import tiktoken
from src.documents import Document


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
            start_char_idx += len(chunk_text)
        return chunks


if __name__ == "__main__":
    chunker = FixedTokenChunker("cl100k_base", max_tokens=5, overlap=0)
    chunks = chunker.chunk_text(
        Chunk(
            doc_id=1,
            text="This is a text that should be split on every five tokens accordingly to the xxx tokenizer."
        )
    )
    print(chunks)
