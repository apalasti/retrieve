from pathlib import Path

import ranx
import orjson
import pandas as pd
from tqdm import tqdm
from src.vector_db import VectorDB


ROOT_DIR = Path(__file__).parent.parent
CORPUS_PATH = ROOT_DIR / "data/msmarco/test_corpus.jsonl"
CUTOFF = 100


def file_len(fpath):
    with open(fpath, "r") as f:
        return sum(1 for _ in f)


def read_jsonl(fpath, callback):
    with open(fpath, "r") as f:
        for line in f:
            yield callback(orjson.loads(line))


def main():
    print("Loading database...")
    db = VectorDB(ROOT_DIR / "data/msmarco/msmarco.db", 0)

    # If db is empty then load the corpus
    if 0 == db.count_rows():
        num_lines = file_len(CORPUS_PATH)
        record_generator = read_jsonl(
            CORPUS_PATH, lambda row: [{"id": row["_id"], "text": row["text"]}]
        )
        db.add(tqdm(record_generator, desc="Loading corpus...", total=num_lines))
    else:
        print("Corpus already loaded!")

    qd_pairs = pd.read_csv(ROOT_DIR / "data/msmarco/qrels/test.tsv", sep="\t")
    qd_pairs.set_index("query-id", inplace=True)

    queries = pd.read_json(ROOT_DIR / "data/msmarco/queries.jsonl", lines=True)
    queries = queries.set_index("_id", drop=False).rename(columns={"_id": "id"})
    queries = queries.loc[qd_pairs.index.unique()]
    queries["text"] = queries["text"].astype("str")

    run = ranx.Run({
        str(query_id): db.bm25_search(query, cutoff=CUTOFF)
        .to_pandas()
        .astype({"id": "str"})
        .set_index("id")["_score"]
        .to_dict()
        for query_id, query in tqdm(
            queries["text"].items(), desc="Running queries", total=len(queries)
        )
    })

    qrels = ranx.Qrels(
        {
            str(query_id): refs.astype({"corpus-id": "str"})
            .set_index("corpus-id")["score"]
            .to_dict()
            for query_id, refs in qd_pairs.groupby("query-id")
        }
    )

    report = ranx.compare(qrels, runs=[run], metrics=["hits", "recall", "precision@5", "precision@10"])
    print(report)


if __name__ == "__main__":
    main()
