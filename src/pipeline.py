from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterator, TypeVar

CacheType = TypeVar("CacheType")
T = TypeVar("T")


def make_pipeline(*transformations: Callable):
    def combined(inputs):
        for t in transformations:
            inputs = t(inputs)
        return inputs

    return combined


class Cache(ABC, Generic[CacheType]):
    @abstractmethod
    def is_fresh(self, key: CacheType):
        raise NotImplementedError()

    @abstractmethod
    def refresh(self, key: CacheType):
        raise NotImplementedError()


class CacheFilter:
    def __init__(self, cache: Cache[T]):
        self.cache = cache

    def __call__(self, inputs: Iterator[T]):
        for input in inputs:
            if not self.cache.is_fresh(input):
                yield input
                self.cache.refresh(input)
