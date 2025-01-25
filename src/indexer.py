import logging
from typing import Callable, List, Iterator

from tqdm import tqdm

from .vector_db import VectorDB
from .documents import Document
from .chunking import Chunk


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
    def __init__(self, vector_store: VectorDB, transformations: List[Callable]) -> None:
        self._vector_store = vector_store
        self._pipeline = _make_pipeline(*transformations)

    def process_documents(self, documents: List[Document], display_progress=False):
        if display_progress:
            with tqdm(documents, desc="Processing documents", total=len(documents)) as pbar:
                for chunks in self._pipeline(self._filter_cached(pbar)):
                    self._vector_store.add_chunks(chunks)
        else:
            for chunks in self._pipeline(self._filter_cached(documents)):
                self._vector_store.add_chunks(chunks)

        self._vector_store._chunks_table.optimize()

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
