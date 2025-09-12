from collections import OrderedDict
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")
# https://gist.github.com/davesteele/44793cd0348f59f8fadd49d7799bd306
class CacheDict(OrderedDict[K, V]):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, *args: Any, cache_len: int = 10, **kwargs: Any):
        assert cache_len > 0
        self.cache_len = cache_len

        super().__init__(*args, **kwargs)

    def __setitem__(self, key: K, value: V) -> None:
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key: K) -> V:
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val

    def set_cache_len(self, new_len: int) -> None:
        """Change the cache length. If new_len is smaller, evict excess LRU items."""
        assert new_len > 0
        self.cache_len = new_len
        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)
