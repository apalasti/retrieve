from pathlib import Path

from watchdog.events import (FileCreatedEvent, FileDeletedEvent,
                             FileModifiedEvent, FileMovedEvent,
                             PatternMatchingEventHandler)
from watchdog.observers import Observer

from .documents import Document, FileDocument
from .indexer import Indexer


class DirectoryWatcher(PatternMatchingEventHandler): 
    def __init__(self, root_dir, indexer: Indexer, patterns: list[str] | None = None):
        super().__init__(patterns=patterns, ignore_directories=True)

        self.root_dir = Path(root_dir)
        self._indexer = indexer

    def start(self) -> None:
        self._observer = Observer()
        self._observer.schedule(self, str(self.root_dir), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()

    def on_created(self, event: FileCreatedEvent) -> None:
        document = FileDocument(event.src_path)
        self._indexer.process_documents([document])

    def on_deleted(self, event: FileDeletedEvent) -> None:
        document = Document(id=event.src_path)
        self._indexer.delete_document(document)

    def on_modified(self, event: FileModifiedEvent) -> None:
        document = FileDocument(event.src_path)
        self._indexer.process_documents([document])

    def on_moved(self, event: FileMovedEvent) -> None:
        self.dispatch(FileDeletedEvent(src_path=event.src_path))
        self.dispatch(FileCreatedEvent(src_path=event.dest_path))
