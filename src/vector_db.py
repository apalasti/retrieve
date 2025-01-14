import lancedb
import lance 
import lancedb.exceptions
import pyarrow as pa

TABLE_NAME = "embeddings"


class VectorDB:
    def __init__(self, uri: str, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.db = lancedb.connect(uri)
        self.setup()

    def setup(self):
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("file_path", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), list_size=self.embedding_dim)),
            pa.field("text", pa.string()),
        ])

        self.table = self.db.create_table(TABLE_NAME, schema=schema, exist_ok=True)
        try:
            self.table.create_fts_index(field_names="text", use_tantivy=False)
        except:
            pass


    def add(self, data):
        return self.table.add(data)

    def count_rows(self):
        return self.table.count_rows()

    def bm25_search(self, query: str, cutoff: int):
        return (
            self.table.search(query=query, query_type="fts", fts_columns="text")
            .limit(cutoff)
            .select(["id", "text"])
        )
