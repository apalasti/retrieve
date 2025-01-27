from typing import List
from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from retrieve.core import (
    VectorDB,
    DirectoryReader,
    STEmbedding,
    Chunk,
    FixedTokenChunker,
)
from retrieve.processing import Indexer
from retrieve.query_engine import QueryEngine

# logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).parent.parent
CORPUS_PATH = ROOT_DIR / "data/general_evaluation_data/corpora"


def evaluate_search(
    query_engine: QueryEngine,
    questions: List[str],
    references: List[List[Chunk]],
    top_k=10,
):
    results = [
        query_engine.search(question, top_k, "hybrid")
        for question in tqdm(
            questions, desc="Performing searches", total=len(questions)
        )
    ]
    evaluation = pd.DataFrame(
        [
            QueryEngine.evaluate_by_overlaps(refs, retrieved)
            for refs, retrieved in zip(references, results)
        ]
    )
    return evaluation


def main():
    print("Loading embedding model...")
    embedding_model = STEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"torch_dtype": "float16"},
    )

    print("Loading database...")
    db = VectorDB(
        CORPUS_PATH / ".vectorstore.db",
        embedding_model.get_embedding_dims(),
    )

    corpus_reader = DirectoryReader(CORPUS_PATH)
    indexer = Indexer(
        db,
        transformations=[
            FixedTokenChunker(max_tokens=250, overlap=125),
            embedding_model,
        ],
    )
    indexer.process_reader(corpus_reader, show_progress=True)
    print(f"Number of embeddings in vector store: {db.num_chunks()}")

    questions_df = pd.read_csv(CORPUS_PATH.parent / "questions_df.csv", sep=",")
    questions_df["question"] = questions_df["question"].astype("str")
    questions_df["references"] = questions_df["references"].apply(orjson.loads)

    corpus_id_to_filepath = {Path(resource).stem: resource for resource in corpus_reader.get_resources()}
    questions_df["corpus_filepath"] = questions_df["corpus_id"].map(corpus_id_to_filepath)

    references = [
        [
            Chunk(
                doc_id=row["corpus_filepath"],
                text=reference["content"],
                metadata={
                    "start_char_idx": reference["start_index"],
                    "end_char_idx": reference["end_index"],
                },
            )
            for reference in row["references"]
        ]
        for _, row in questions_df.iterrows()
    ]

    query_engine = QueryEngine(db, embedding_model)
    k_values = [1, 3, 5, 10]
    all_evaluations = []
    for k in k_values:
        evaluation_k = evaluate_search(
            query_engine, questions_df["question"].to_list(), references, top_k=k
        )
        evaluation_k["top_k"] = k  # Add a column to indicate top_k
        all_evaluations.append(evaluation_k)
    combined_evaluation = pd.concat(all_evaluations)
    summary = combined_evaluation.groupby("top_k").agg(["mean", "std", "median"])
    print(summary)


if __name__ == "__main__":
    main()
