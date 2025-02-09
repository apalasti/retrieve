import logging
import time
from typing import Callable, Iterator, List, Union

from tqdm import tqdm

from ..core import Chunk, Document, DocumentReader, VectorDB

logger = logging.getLogger(__name__)


def _make_pipeline(
    *transformations: Callable,
) -> Callable[[Iterator[Document]], Iterator[Chunk]]:
    def combined(inputs):
        for t in transformations:
            inputs = t(inputs)
        return inputs

    return combined


class Indexer:
    def __init__(
        self, vector_store: VectorDB, transformations: List[Callable], cache=True
    ) -> None:
        self._vector_store = vector_store
        self._pipeline = _make_pipeline(*transformations)
        self.cache = cache

    def apply_transformations(self, reader: Union[DocumentReader | List[Document]]):
        if isinstance(reader, DocumentReader):
            docs = reader.iter_documents()
            num_documents = reader.num_documents()
        else:
            docs = reader
            num_documents = len(reader)

        results = []
        with tqdm(docs, desc="Processing documents", total=num_documents) as pbar:
            for chunks in self._pipeline(pbar):
                if isinstance(chunks, List):
                    results.extend(chunks)
                else:
                    results.append(chunks)
        return results

    def _process(self, iterator: Iterator[Document], length: int, show_progress: bool):
        pipeline = (
            (lambda inputs: self._pipeline(self._filter_cached(inputs)))
            if self.cache
            else self._pipeline
        )

        if show_progress:
            with tqdm(
                iterator, desc="Processing documents", total=length, unit="doc"
            ) as pbar:
                inserted = 0
                for chunks in pipeline(pbar):
                    self._vector_store.add_chunks(chunks)
                    inserted += len(chunks)
                    pbar.set_postfix({"inserted": inserted})
        else:
            for chunks in pipeline(iterator):
                self._vector_store.add_chunks(chunks)

        start = time.perf_counter_ns()
        self._vector_store._chunks_table.optimize()
        end = time.perf_counter_ns()
        logger.info(f"Optimizing table, took: {(end-start)* 1e-9:.4f} seconds")

    def process_documents(self, documents: List[Document], show_progress=False):
        self._process(iter(documents), len(documents), show_progress)

    def process_reader(self, reader: DocumentReader, show_progress=False):
        self._process(reader.iter_documents(), reader.num_documents(), show_progress)

    def delete_document(self, document: Document, references=True):
        logger.info("Deleting document: %s", document)
        self._vector_store.delete_document(document.id, references)

    def _filter_cached(self, documents: Iterator[Document]):
        for document in documents:
            if self._vector_store.refresh_document(document):
                logger.info("Processing document: %s", document)
                yield document
            else:
                logger.info("Document already processed: %s", document)
