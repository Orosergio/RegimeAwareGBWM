"""A tiny generic registry used to make policies/agents config-addressable.

Example
-------
>>> from gbwm.registry import Registry
>>> policies = Registry("policy")
>>> @policies.register("buy_and_hold")
... class BuyAndHold: ...
>>> policies.create("buy_and_hold", ...)
"""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            key = name.lower()
            if key in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[key] = cls
            return cls

        return deco

    def get(self, name: str) -> type[T]:
        key = name.lower()
        if key not in self._items:
            raise KeyError(
                f"unknown {self.kind} '{name}'. available: {sorted(self._items)}"
            )
        return self._items[key]

    def create(self, name: str, *args, **kwargs) -> T:
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items
