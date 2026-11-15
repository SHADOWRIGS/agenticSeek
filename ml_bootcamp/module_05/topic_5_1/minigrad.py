"""Урок 5.1.1: автоматическое дифференцирование в обратном режиме на NumPy — учебная реализация."""

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Сворачивает градиент к форме операнда, который был растянут правилами broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, parents: tuple["Tensor", ...] = ()) -> None:
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._backward = lambda: None  # локальное правило: как раздать self.grad родителям

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def _unary(self, value: np.ndarray, local_grad: np.ndarray) -> "Tensor":
        out = Tensor(value, (self,))

        def _backward():
            self.grad += out.grad * local_grad

        out._backward = _backward
        return out

    def relu(self):
        return self._unary(np.maximum(self.data, 0.0), (self.data > 0).astype(float))

    def exp(self):
        value = np.exp(self.data)
        return self._unary(value, value)

    def log(self):
        return self._unary(np.log(self.data), 1.0 / self.data)

    def sum(self, axis: int | None = None):
        out = Tensor(self.data.sum(axis=axis, keepdims=axis is not None), (self,))

        def _backward():
            self.grad += np.broadcast_to(out.grad, self.data.shape)

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def backward(self) -> None:
        order: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: "Tensor") -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._parents:
                    visit(parent)
                order.append(node)  # узел попадает в список после всех своих родителей

        visit(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            node._backward()
