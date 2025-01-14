from typing import Literal
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT_DIR = Path(__file__).parent.parent
NUM_DOCUMENTS = 1_000_000
DATASET = "test"


def msmarco_searched_documents(
    location: Path | str, dataset: Literal["train", "test", "dev"]
):
    location = Path(location)
    df = pd.read_csv(location / f"{dataset}.tsv", sep="\t")
    return df["corpus-id"].astype("int32").unique()


def file_len(fpath):
    with open(fpath, "r") as f:
        return sum(1 for _ in f)


def main():
    msmarco_dir = ROOT_DIR / "data/msmarco"
    test_docs = msmarco_searched_documents(msmarco_dir / "qrels", DATASET)

    doc_ids = np.setdiff1d(np.arange(0, 8_841_822), test_docs)

    np.random.seed(2)
    np.random.shuffle(doc_ids)

    filtered_ids = np.concatenate([test_docs, doc_ids[:NUM_DOCUMENTS]])
    filtered_ids = set(filtered_ids)

    chunks = pd.read_json(msmarco_dir / "corpus.jsonl", lines=True, chunksize=100_000)
    num_chunks = file_len(msmarco_dir / "corpus.jsonl") / 100_000
    with open(msmarco_dir / f"{DATASET}_corpus.jsonl", "w") as f:
        for df in tqdm(chunks, desc="Filtering corpus", total=num_chunks, unit="chunk"):
            df.set_index("_id", inplace=True, drop=False)
            df.drop(["title", "metadata"], axis=1, inplace=True)
            df.loc[df.index.isin(filtered_ids)].to_json(f, orient="records", lines=True)


if __name__ == "__main__":
    main()
