from pathlib import Path

import pandas as pd
from tqdm import tqdm

from retrieve.core import FixedTokenChunker, JsonLReader, LlamaEmbedding, Embedder, VectorDB
from retrieve.processing import Indexer
from retrieve.query_engine import QueryEngine
from retrieve.reranking import LBReranker

ROOT_DIR = Path(__file__).parent.parent
CORPUS_PATH = ROOT_DIR / "data/msmarco/test_corpus.jsonl"

SEARCH_TYPE = "hybrid"


def file_len(fpath):
    with open(fpath, "r") as f:
        return sum(1 for _ in f)


def main():
    print("Loading embedding model...")
    embedder = LlamaEmbedding(
        repo_id="second-state/All-MiniLM-L6-v2-Embedding-GGUF",
        file_name="all-MiniLM-L6-v2-Q8_0.gguf"
    )

    print("Loading database...")
    db = VectorDB(
        ROOT_DIR / "data/msmarco/msmarco.db",
        embedder.get_embedding_dims(),
    )

    # If db is empty then load the corpus
    if 0 == db.num_chunks():
        Embedder.EMBEDDING_BATCH_SIZE = 10_000
        corpus_reader = JsonLReader(
            CORPUS_PATH,
            lambda record: {
                "id": record["_id"],
                "text": record["text"],
            },
        )
        indexer = Indexer(
            db, transformations=[FixedTokenChunker(max_tokens=250, overlap=125), embedder],
            cache=False,
        )
        indexer.process_reader(corpus_reader, show_progress=True)

    print(f"Number of embeddings in vector store: {db.num_chunks()}")
    query_engine = QueryEngine(db, embedder, LBReranker(n_ctx=1000))

    qd_pairs = pd.read_csv(ROOT_DIR / "data/msmarco/qrels/test.tsv", sep="\t")
    qd_pairs.set_index("query-id", inplace=True)
    qrels = {
        str(query_id): refs.astype({"corpus-id": "str"})
        .set_index("corpus-id")["score"]
        .to_dict()
        for query_id, refs in qd_pairs.groupby("query-id")
    }

    queries = pd.read_json(ROOT_DIR / "data/msmarco/queries.jsonl", lines=True)
    queries = queries.set_index("_id", drop=False).rename(columns={"_id": "id"})
    queries = queries.loc[qd_pairs.index.unique()]
    queries["text"] = queries["text"].astype("str")

    evaluations = []
    for query_id, query_text in tqdm(
        queries["text"].items(), desc="Running queries", total=len(queries)
    ):
        retrieved = query_engine.search(query_text, k=100, type=SEARCH_TYPE)
        retrieved.sort(key=lambda c: c.metadata["score"], reverse=True)
        for top_k in [1, 5, 10, 50]:
            assert top_k == len(retrieved[:top_k])
            evaluation = query_engine.evaluate_by_relevance(
                qrels[str(query_id)], retrieved[:top_k], ["precision", "ndcg"]
            )
            evaluation["query_id"] = query_id
            evaluation["top_k"] = top_k
            evaluations.append(evaluation)

    summary = pd.DataFrame(evaluations)
    print(summary.groupby("top_k")[["precision", "ndcg"]].describe())


if __name__ == "__main__":
    main()
