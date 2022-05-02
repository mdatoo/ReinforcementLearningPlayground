from typing import Generic, TypeVar

A = TypeVar('A')
B = TypeVar('B')


class ActionSelector(Generic[A, B]):
    def select(self, values: A) -> B:
        raise NotImplementedError

    def step(self) -> None:
        pass
