"""Практика 4.2.4: случайный лес для регрессии поверх готового дерева.

Реализуйте функции и методы, в которых стоит raise NotImplementedError. Ожидаемое поведение описано
в docstring и проверяется тестами: uv run pytest -q

Из scikit-learn в этом модуле разрешён только sklearn.tree.DecisionTreeRegressor.
Во всех функциях X — таблица признаков формы (n, p): массив NumPy или DataFrame; внутри она
приводится к массиву вещественных чисел. Входные данные изменять нельзя.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor  # noqa: F401  (понадобится в fit)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Коэффициент детерминации: 1 - сумма квадратов ошибок / сумма квадратов отклонений y_true от среднего.

    Если все значения y_true одинаковы, знаменатель равен нулю: в этом случае возвращается 1.0
    при точном совпадении предсказаний с y_true и 0.0 иначе.
    """
    raise NotImplementedError


class MyRandomForestRegressor:
    """Случайный лес для регрессии: бутстреп-выборки, деревья DecisionTreeRegressor, усреднение.

    Параметры:
        n_estimators — число деревьев;
        max_features — сколько признаков рассматривать в каждом узле; передаётся в дерево без изменений;
        max_depth, min_samples_leaf — передаются в дерево без изменений;
        bootstrap — True: каждое дерево учится на выборке n объектов с возвращением;
                    False: каждое дерево учится на всех объектах в исходном порядке;
        random_state — целое число или None.

    Случайность. Каждое дерево получает собственный независимый генератор:

        seeds = np.random.SeedSequence(random_state).spawn(n_estimators)
        rng = np.random.default_rng(seeds[b])          # генератор дерева номер b

    Из rng сначала берутся номера объектов бутстреп-выборки (rng.integers(0, n, size=n)),
    затем random_state для дерева (int(rng.integers(0, 2**31 - 1))). Следствия, которые проверяют тесты:
    одинаковый random_state даёт одинаковый лес, а дерево номер b не зависит от n_estimators.

    После fit доступны атрибуты:
        estimators_ — список обученных деревьев;
        estimators_samples_ — список массивов длины n: номера объектов, на которых училось каждое дерево
                              (при bootstrap=False — np.arange(n));
        n_features_in_ — число признаков;
        oob_prediction_ — массив длины n: для каждого объекта среднее предсказание тех деревьев,
                          в обучающую выборку которых он не попал; nan, если таких деревьев нет;
        oob_score_ — r2_score по объектам, у которых oob_prediction_ не nan; nan, если таких объектов нет.

    Предупреждений NumPy (например, о среднем пустого массива) возникать не должно.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: int | float | str | None = 1.0,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MyRandomForestRegressor":
        """Обучает лес и считает out-of-bag оценку. Возвращает self.

        Требование к скорости: OOB-предсказания считаются по одному вызову predict на дерево
        (для всех его OOB-объектов сразу), а не отдельным вызовом для каждого объекта.
        """
        raise NotImplementedError

    def tree_predictions(self, X: np.ndarray) -> np.ndarray:
        """Предсказания отдельных деревьев: массив формы (n_estimators, n), строка b — дерево номер b."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание леса — среднее предсказаний деревьев. Массив формы (n,)."""
        raise NotImplementedError


def permutation_importance(
    model, X: np.ndarray, y: np.ndarray, n_repeats: int = 5, random_state: int | None = None
) -> np.ndarray:
    """Перестановочная важность признаков по метрике R².

    model — любой объект с методом predict(X).
    Возвращает массив формы (p, n_repeats): элемент [j, r] равен
    r2_score(y, model.predict(X)) - r2_score(y, model.predict(X_perm)),
    где X_perm — копия X, в которой значения столбца j случайно переставлены между объектами,
    а остальные столбцы не тронуты. Для каждого признака и каждого повтора перестановка новая.

    Случайность: один генератор np.random.default_rng(random_state) на весь вызов.
    Модель заново не обучается.
    """
    raise NotImplementedError


def partial_dependence(model, X: np.ndarray, feature: int, grid: np.ndarray) -> np.ndarray:
    """Частичная зависимость предсказания модели от признака с номером feature.

    Для каждого значения v из grid: во всех объектах X значение признака заменяется на v,
    и предсказания модели усредняются по объектам. Возвращает массив формы (len(grid),).
    """
    raise NotImplementedError
