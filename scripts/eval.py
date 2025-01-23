from pathlib import Path

import ranx
import pandas as pd
from tqdm import tqdm

from src.vector_db import VectorDB
from src.embedding import STEmbedding
from src.chunking import FixedTokenChunker
from src.documents import JsonLReader
from src.pipeline import make_pipeline


ROOT_DIR = Path(__file__).parent.parent
CORPUS_PATH = ROOT_DIR / "data/msmarco/test_corpus.jsonl"
WORD_LIMIT_PER_PASSAGE = 100
CUTOFF = 100


class BM25Search:
    name: str = "BM25"

    def __call__(self, db: VectorDB, query):
        return db.bm25_search(query["text"], cutoff=CUTOFF)


class VectorSearch:
    name = "vector-search"

    def __call__(self, db: VectorDB, query):
        result = db.vector_search(query["embedding"], cutoff=CUTOFF)
        result["_score"] = 1 - result["_distance"]
        return result


def file_len(fpath):
    with open(fpath, "r") as f:
        return sum(1 for _ in f)


def main():
    print("Loading embedding model...")
    embedder = STEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"torch_dtype": "float16"},
    )

    print("Loading database...")
    db = VectorDB(
        ROOT_DIR / "data/msmarco/msmarco.db",
        embedder.get_embedding_dims(),
    )

    # If db is empty then load the corpus
    if 0 == db.num_chunks():
        corpus_reader = JsonLReader(
            CORPUS_PATH,
            lambda record: {
                "id": record["_id"],
                "text": record["text"],
                "size": len(record["text"]),
            },
        )
        pipeline = make_pipeline(FixedTokenChunker(max_tokens=100, overlap=0), embedder)
        for processed_batch in pipeline.run(
            tqdm(corpus_reader.iter_documents(), total=corpus_reader.num_documents())
        ):
            db.add_chunks(processed_batch)

        print("Optimizing table")
        db.chunks_table.optimize()
    else:
        print("Corpus already loaded!")

    qd_pairs = pd.read_csv(ROOT_DIR / "data/msmarco/qrels/test.tsv", sep="\t")
    qd_pairs.set_index("query-id", inplace=True)

    queries = pd.read_json(ROOT_DIR / "data/msmarco/queries.jsonl", lines=True)
    queries = queries.set_index("_id", drop=False).rename(columns={"_id": "id"})
    queries = queries.loc[qd_pairs.index.unique()]
    queries["text"] = queries["text"].astype("str")
    queries["embedding"] = embedder.embed_texts(queries["text"].values)

    search_methods = [BM25Search(), VectorSearch()]

    runs = []
    for method in search_methods:
        with tqdm(
            queries.index,
            desc=f"Running queries with {method.name}",
            leave=True,
            unit="query",
        ) as pbar:
            run_dict = {
                str(query_id): (df := method(db, queries.loc[query_id]))
                .set_index(df.index.astype("str"))["_score"]
                .to_dict()
                for query_id in pbar
            }
        runs.append(ranx.Run(run_dict, method.name))

    qrels = ranx.Qrels(
        {
            str(query_id): refs.astype({"corpus-id": "str"})
            .set_index("corpus-id")["score"]
            .to_dict()
            for query_id, refs in qd_pairs.groupby("query-id")
        }
    )

    report = ranx.compare(
        qrels, runs, metrics=["hits", "recall", "precision", "precision@5", "precision@10"]
    )
    print(report)


if __name__ == "__main__":
    main()
