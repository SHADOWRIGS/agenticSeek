"""Практика 4.1.4: дерево решений для классификации с нуля на NumPy.

Реализуйте функции и методы, в которых стоит raise NotImplementedError. Ожидаемое поведение описано
в docstring и проверяется тестами: uv run pytest -q

Использовать scikit-learn в этом модуле нельзя.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """Узел дерева. Этот класс готов, менять его не нужно.

    value — доли классов среди объектов узла, в порядке classes_ дерева.
    У листа feature, threshold, left и right равны None.
    Объект идёт в левого потомка, если X[feature] <= threshold, иначе в правого.
    """

    value: np.ndarray
    n_samples: int
    impurity: float
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def impurity(counts: np.ndarray, criterion: str = "gini") -> float:
    """Неоднородность узла по числу объектов каждого класса.

    counts — одномерный массив неотрицательных чисел: сколько объектов каждого класса в узле.
    criterion: "gini" — индекс Джини, "entropy" — энтропия с логарифмом по основанию 2.
    Классы с нулевым числом объектов не должны приводить к nan. Для пустого узла возвращается 0.0.
    При неизвестном критерии — ValueError.
    """
    raise NotImplementedError


def best_split(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    criterion: str = "gini",
    min_samples_leaf: int = 1,
) -> tuple[int, float, float] | None:
    """Лучшее разбиение узла.

    X — признаки объектов узла, форма (n, p). y — номера классов от 0 до n_classes - 1, форма (n,).

    Кандидаты в пороги для признака — середины между соседними различными значениями этого признака.
    Допустимо только разбиение, после которого в каждом потомке не меньше min_samples_leaf объектов.

    Возвращает кортеж (feature, threshold, gain), где gain — уменьшение неоднородности:
    impurity(узел) - n_left / n * impurity(левый) - n_right / n * impurity(правый).

    Выбирается разбиение с наибольшим gain; нулевой gain допустим. Разбиения, у которых gain
    отличается меньше чем на 1e-12, считаются равными: из них выбирается разбиение с меньшим
    номером признака, а при одном признаке — с меньшим порогом.

    Если допустимых разбиений нет (например, все объекты одинаковы), возвращает None.

    Требование к скорости: один проход по отсортированным значениям признака с накопленными
    счётчиками классов. Пересчёт неоднородности с нуля для каждого порога не пройдёт тест времени.
    """
    raise NotImplementedError


class MyDecisionTreeClassifier:
    """Дерево решений для классификации.

    Параметры:
        max_depth — максимальная глубина; None — без ограничения. Корень имеет глубину 0.
        min_samples_split — узел с меньшим числом объектов не делится.
        min_samples_leaf — минимальное число объектов в каждом потомке.
        criterion — "gini" или "entropy".

    Узел становится листом, если: все его объекты одного класса; достигнута max_depth;
    объектов меньше min_samples_split; best_split вернул None.

    После fit доступны атрибуты:
        classes_ — отсортированные уникальные метки классов;
        n_features_in_ — число признаков;
        root_ — корневой Node.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MyDecisionTreeClassifier":
        """Строит дерево. Метки y могут быть любыми: числа, строки. Возвращает self."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Доли классов в листе, куда попал объект. Форма (n, число классов), столбцы в порядке classes_."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Метка самого частого класса листа. При равенстве долей — класс с меньшим номером в classes_."""
        raise NotImplementedError

    def get_depth(self) -> int:
        """Глубина дерева: число разбиений на самом длинном пути от корня до листа."""
        raise NotImplementedError

    def get_n_leaves(self) -> int:
        """Число листьев."""
        raise NotImplementedError
