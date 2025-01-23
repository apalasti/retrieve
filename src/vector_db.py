from typing import List

import lancedb
import numpy as np
import orjson
import pandas as pd
import pyarrow as pa

from src.chunking import Chunk
from src.documents import FileDocument
from src.pipeline import Cache

CHUNKS_TABLE = "chunks"
DOCUMENTS_TABLE = "documents"


class VectorDB(Cache[FileDocument]):
    def __init__(self, uri: str, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.db = lancedb.connect(uri)
        self.setup()

    def setup(self):
        # Create tables no problem if they already exist
        self.chunks_table = self.db.create_table(
            CHUNKS_TABLE, schema=self._chunks_schema(), exist_ok=True
        )
        self.docs_table = self.db.create_table(
            DOCUMENTS_TABLE, schema=self._docs_schema(), exist_ok=True
        )

        try:
            self.chunks_table.create_fts_index(field_names="text", use_tantivy=False)
        except:
            pass

    def _docs_schema(self):
        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("file_path", pa.string(), nullable=False),
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
            self.chunks_table.search()
            .where(f"doc_id = '{doc_id}'")
            .select(["text", "metadata"])
            .to_pandas()
        )
        return chunks

    def add_chunks(self, chunks: List[Chunk]):
        self.chunks_table.add([chunk.to_dict() for chunk in chunks])

    def delete_chunks(self, doc_id: str):
        self.chunks_table.delete(f"doc_id = '{doc_id}'")

    def num_chunks(self):
        return self.chunks_table.count_rows()

    def get_doc(self, file_path: str):
        return (
            self.docs_table.search()
            .where(f"file_path = '{file_path}'")
            .limit(1)
            .to_list()
            or [None]
        )[0]

    def is_fresh(self, doc: FileDocument):
        file_path = doc.metadata["full_path"]
        doc_hash = self._file_hash(doc)
        cached_doc = self.get_doc(file_path)
        return cached_doc is not None and cached_doc["hash"] == doc_hash

    def refresh(self, doc: FileDocument):
        file_path = doc.metadata["full_path"]
        doc_hash = self._file_hash(doc)
        cached_doc = self.get_doc(file_path)

        cached_doc = self.get_doc(file_path)
        if cached_doc is None:
            self.docs_table.add([
                {
                    "id": doc.id,
                    "file_path": file_path,
                    "metadata": orjson.dumps(doc.metadata),
                    "hash": doc_hash,
                }
            ])
        elif doc_hash != cached_doc["hash"]:
            self.update_doc(file_path, doc)

    def update_doc(self, file_path: str, doc: FileDocument):
        doc_hash = self._file_hash(doc)
        self.docs_table.update(
            f"file_path = '{file_path}'",
            {
                "file_path": doc.metadata["full_path"],
                "metadata": orjson.dumps(doc.metadata),
                "hash": doc_hash,
            },
        )

    def delete_doc(self, file_path: str):
        doc = self.get_doc(file_path)
        if doc is not None:
            self.docs_table.delete(f"id = '{doc['id']}'")
            self.chunks_table.delete(f"doc_id = '{doc['id']}'")

    def _file_hash(self, doc: FileDocument):
        file_path = doc.metadata["full_path"]
        modified_time = doc.metadata["modified_time"]
        return f"{file_path}:{modified_time}"

    def bm25_search(self, query: str, cutoff: int):
        result: pd.DataFrame = (
            self.chunks_table.search(query=query, query_type="fts", fts_columns="text")
            .limit(cutoff)
            .select(["doc_id", "text", "metadata"])
            .to_pandas()
        )
        result.set_index("doc_id", inplace=True)
        return result

    def vector_search(self, vector: np.ndarray, cutoff: int):
        result: pd.DataFrame = (
            self.chunks_table.search(vector, vector_column_name="embedding")
            .metric("dot")
            .limit(cutoff)
            .select(["doc_id", "text", "metadata"])
            .to_pandas()
        )
        result.set_index("doc_id", inplace=True)
        return result
