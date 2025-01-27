from typing import List

import lancedb
import numpy as np
import orjson
import pandas as pd
import pyarrow as pa

from .chunking import Chunk
from .documents import Document

CHUNKS_TABLE = "chunks"
DOCUMENTS_TABLE = "documents"


class VectorDB:
    def __init__(self, uri: str, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.db = lancedb.connect(uri)
        self.setup()

    def setup(self):
        # Create tables no problem if they already exist
        self._chunks_table = self.db.create_table(
            CHUNKS_TABLE, schema=self._chunks_schema(), exist_ok=True
        )
        self._docs_table = self.db.create_table(
            DOCUMENTS_TABLE, schema=self._docs_schema(), exist_ok=True
        )

        try:
            self._chunks_table.create_fts_index(field_names="text", use_tantivy=False)
        except:
            pass

    def _docs_schema(self):
        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("metadata", pa.json_(), nullable=True),
                pa.field("hash", pa.string(), nullable=False),
            ]
        )

    def _chunks_schema(self):
        return pa.schema(
            [
                pa.field("doc_id", pa.string(), nullable=False),
                pa.field("text", pa.string(), nullable=False),
                pa.field(
                    "embedding",
                    pa.list_(pa.float32(), list_size=self.embedding_dim),
                    nullable=False,
                ),
                pa.field("metadata", pa.json_(), nullable=True),
            ]
        )

    def get_chunks(self, doc_id: int) -> pd.DataFrame:
        """The text chunks in order already embedded in the file."""
        chunks: pd.DataFrame = (
            self._chunks_table.search()
            .where(f"doc_id = '{doc_id}'")
            .select(["text", "metadata"])
            .to_pandas()
        )
        return chunks

    def add_chunks(self, chunks: List[Chunk]):
        self._chunks_table.add([chunk.to_dict() for chunk in chunks])

    def delete_chunks(self, doc_id: str):
        self._chunks_table.delete(f"doc_id = '{doc_id}'")

    def num_chunks(self):
        return self._chunks_table.count_rows()

    def get_document(self, id: str):
        return (
            self._docs_table.search()
            .where(f"id = '{id}'")
            .limit(1)
            .to_list()
            or [None]
        )[0]

    def refresh_document(self, document: Document):
        cached_doc = self.get_document(document.id)
        if cached_doc is None:
            self._docs_table.add([
                {
                    "id": document.id,
                    "metadata": orjson.dumps(document.metadata),
                    "hash": document.hash(),
                }
            ])
            return True

        if cached_doc["hash"] != document.hash():
            self._docs_table.update(
                f"id = '{document.id}'",
                {"metadata": orjson.dumps(document.metadata), "hash": document.hash()},
            )
            return True

        return False

    def delete_document(self, id: str, references=True):
        self._docs_table.delete(f"id = '{id}'")
        if references:
            self._chunks_table.delete(f"doc_id = '{id}'")

    def bm25_search(self, query: str, cutoff: int):
        result: pd.DataFrame = (
            self._chunks_table.search(query=query, query_type="fts", fts_columns="text")
            .limit(cutoff)
            .select(["doc_id", "text", "metadata"])
            .to_pandas()
        )
        result.set_index("doc_id", inplace=True)
        return result

    def vector_search(self, vector: np.ndarray, cutoff: int):
        result: pd.DataFrame = (
            self._chunks_table.search(vector, vector_column_name="embedding")
            .metric("dot")
            .limit(cutoff)
            .select(["doc_id", "text", "metadata"])
            .to_pandas()
        )
        result.set_index("doc_id", inplace=True)
        return result
