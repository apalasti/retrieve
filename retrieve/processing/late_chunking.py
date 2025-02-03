from typing import List, Tuple, Iterator

import numpy as np

from retrieve.core import Chunker, Embedder, Document, Chunk


class LateChunking(Chunker):
    def __init__(self, chunker: Chunker, embedding_model: Embedder) -> None:
        self.chunker = chunker
        self.embedding_model = embedding_model

    def get_chunk_boundaries(self, text: str) -> List[Tuple[int]]:
        return self.chunker.get_chunk_boundaries(text)

    def chunk_node(self, node: Document | Chunk) -> List[Chunk]:
        assert node.embedding is not None
        assert node.embedding.ndim == 2

        tokenization = self.embedding_model.tokenize(node.read())
        token_offsets = np.array(
            [token_end for _, token_end in tokenization["offset_mapping"]]
        )
        assert (
            token_offsets.shape[0] == node.embedding.shape[0]
        ), f"Number of tokens: {len(token_offsets)} != Number of embeddings: {node.embedding.shape[0]}"

        start_offset = node.metadata.get("start_char_idx", 0)
        def char2token(char_idx):
            return token_offsets.searchsorted(char_idx - start_offset, side="right")

        chunks = self.chunker.chunk_node(node)
        for chunk in chunks:
            start = char2token(chunk.metadata["start_char_idx"])
            end = char2token(chunk.metadata["end_char_idx"])
            embedding = np.array(node.embedding[start:end].cpu(), dtype=np.float32).mean(axis=0)
            chunk.embedding = embedding / np.linalg.norm(embedding)
        return chunks

    def __call__(self, nodes: Iterator[Document | Chunk]):
        for nodes_ in self.embedding_model(nodes, output_value="token_embeddings"):
            for node in nodes_:
                yield self.chunk_node(node)
