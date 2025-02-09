from typing import List, Literal, Dict, Optional

import ranx

from .core import Chunk, Embedder, VectorDB
from .reranking import Reranker


def sum_of_ranges(*ranges):
    return sum(end - start for start, end in ranges)


def intersect_ranges(range1, range2):
    # Unpack the ranges
    start1, end1 = range1
    start2, end2 = range2

    # Calculate the maximum of the starting indices and the minimum of the ending indices
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)

    # Check if the intersection is valid (the start is less than or equal to the end)
    if intersect_start <= intersect_end:
        return (intersect_start, intersect_end)
    else:
        return None  # Return an None if there is no intersection


class QueryEngine:

    def __init__(
        self,
        db: VectorDB,
        embedding_model: Embedder,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self._db = db
        self._embedding_model = embedding_model
        self._reranker = reranker

    def search(self, query_text: str, k=10, type: Literal["fts", "vector", "hybrid"] = "hybrid") -> List[Chunk]:
        if type == "fts":
            query = self._db._chunks_table.search(query_text, query_type=type)
        elif type == "vector":
            query_vector = self._embedding_model.embed_texts([query_text])[0]
            query = self._db._chunks_table.search(query_vector, query_type=type).metric(
                "dot"
            )
        elif type == "hybrid":
            query_vector = self._embedding_model.embed_texts([query_text])[0]
            query = (
                self._db._chunks_table.search(query_type=type)
                .vector(query_vector)
                .text(query_text)
                .metric("dot")
            )

        search_results = []
        for chunk_dict in query.limit(k).to_list():
            chunk = Chunk.from_dict(chunk_dict)
            if "_score" in chunk_dict:
                chunk.metadata["score"] = chunk_dict["_score"]
            elif "_relevance_score" in chunk_dict:
                chunk.metadata["score"] = chunk_dict["_relevance_score"]
            elif "_distance" in chunk_dict:
                chunk.metadata["score"] = 1 - chunk_dict["_distance"]
            search_results.append(chunk)
        search_results = Chunk.merge_chunks(*search_results)
        if self._reranker:
            self._reranker.rerank(query_text, search_results)
        return search_results

    @staticmethod
    def evaluate_by_overlaps(references: List[Chunk], retrieved: List[Chunk]):
        retrieved = Chunk.merge_chunks(*retrieved)

        def extract_range(chunk: Chunk):
            return (chunk.metadata["start_char_idx"], chunk.metadata["end_char_idx"])

        overlaps = []
        for reference in references:
            ref_range = extract_range(reference)
            for chunk in retrieved:
                if reference.doc_id != chunk.doc_id:
                    continue

                chunk_range = extract_range(chunk)
                intersection = intersect_ranges(ref_range, chunk_range)
                if intersection is not None:
                    overlaps.append(intersection)

        # Character counts
        n_overlaps = sum_of_ranges(*overlaps)
        n_references = sum_of_ranges(*map(extract_range, references))
        n_retrieved = sum_of_ranges(*map(extract_range, retrieved))

        return {
            "recall": n_overlaps / n_references,
            "precision": n_overlaps / n_retrieved,
            "IoU": n_overlaps / (n_references + n_retrieved - n_overlaps),
        }

    @staticmethod
    def evaluate_by_relevance(qrels: Dict[str, float], retrieved: List[Chunk], metrics: List[str]):
        run = {
            "q_1": {chunk.doc_id: chunk.metadata.get("score", 1) for chunk in retrieved}
        }
        return ranx.evaluate({"q_1": qrels}, run, metrics)
