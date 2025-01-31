import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional

import orjson


@dataclass
class Document:
    id: str
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

    def read(self) -> str:
        return self.text

    def hash(self) -> str:
        return f"{self.id}:{self.read()}"

    def __repr__(self) -> str:
        max_length = 30
        truncated_text = (
            (self.text[:max_length] + "...")
            if self.text is not None and len(self.text) > max_length
            else self.text
        )
        return f"{self.__class__.__name__}(id={self.id}, text='{truncated_text}', metadata={self.metadata})"


class FileDocument(Document):
    def __init__(self, path: str):
        fpath = Path(path).absolute()
        if not fpath.is_file():
            raise RuntimeError(f"Given path does not point to file: {fpath}")

        stats = fpath.stat()
        metadata = {
            "full_path": str(fpath),
            "extension": fpath.suffix,
            "modified_time": stats.st_mtime_ns,
            "size": stats.st_size,
        }

        super().__init__(id=str(fpath), metadata=metadata, text=None)

    def read(self):
        if self.text is None:
            with open(self.metadata["full_path"], "r", errors='ignore') as f:
                self.text = f.read()
        return self.text

    def hash(self):
        return f"{self.metadata['full_path']}:{self.metadata['modified_time']}"


class DocumentReader(ABC):
    """Abstract class defining an object which returns an iterator over documents."""

    @abstractmethod
    def iter_documents(self) -> Generator[Document, Any, None]:
        raise NotImplementedError()

    def load_documents(self):
        return list(self.iter_documents())

    def num_documents(self) -> int:
        return sum(1 for _ in self.iter_documents())


class DirectoryReader(DocumentReader):
    def __init__(self, directory: str, recursive = False, ignore_hidden = True) -> None:
        self.dir_path: Path = Path(directory)
        self.ignore_hidden = ignore_hidden
        if not self.dir_path.is_dir():
            raise RuntimeError(
                f"Given path does not point to directory: {self.dir_path}"
            )

        self.recursive = recursive

    def get_resources(self):
        file_iterator = (
            self.dir_path.rglob("*") if self.recursive else self.dir_path.glob("*")
        )
        return [
            str(f)
            for f in file_iterator
            if f.is_file() and (not any(part.startswith(".") for part in f.parts))
        ]

    def num_documents(self):
        return len(self.get_resources())

    def iter_documents(self):
        for resource_path in self.get_resources():
            yield FileDocument(resource_path)


class JsonLReader(DocumentReader):
    def __init__(self, fpath: str, transform: Callable[[Dict], Dict]):
        self.file_path = Path(fpath)
        self.transform = transform

    def num_documents(self) -> int:
        with open(self.file_path, "r") as f:
            doc_count = sum(1 for _ in f)
        return doc_count

    def iter_documents(self):
        with open(self.file_path, "r") as f:
            for line in f:
                yield Document(**self.transform(orjson.loads(line)))
