"""Практика 3.4.4: рекомендательная система интернет-магазина.

Реализуйте функции и классы этого модуля. Ожидаемое поведение описано в docstring
и проверяется тестами: uv run pytest -q

Из двух персональных моделей — ItemKNNRecommender и ALSRecommender — обязательна одна, на ваш выбор.
Тесты нереализованной модели будут пропущены.
"""

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

DEFAULT_WEIGHTS = {"view": 1.0, "purchase": 3.0}


def time_split(events: pd.DataFrame, split_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит журнал событий по времени. Исходная таблица не изменяется.

    Возвращает (train, test): в train — события с timestamp строго раньше split_date,
    в test — все остальные (событие ровно в момент split_date относится к test).
    Столбцы и порядок строк внутри каждой части сохраняются.
    """
    raise NotImplementedError


def build_interaction_matrix(
    events: pd.DataFrame,
    weights: dict[str, float] | None = None,
    user_ids: np.ndarray | None = None,
    item_ids: np.ndarray | None = None,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Строит разреженную матрицу «пользователь × товар» и отображения индекс → идентификатор.

    Возвращает (matrix, user_ids, item_ids):

    - user_ids и item_ids — массивы идентификаторов: строка i матрицы соответствует пользователю
      user_ids[i], столбец j — товару item_ids[j]. Если аргументы user_ids / item_ids не заданы,
      это отсортированные по возрастанию уникальные идентификаторы из events. Если заданы — матрица
      строится в этой индексации (так тестовая матрица согласуется с обучающей), а события
      с неизвестными пользователями или товарами отбрасываются;
    - matrix — scipy.sparse.csr_matrix типа float размера (len(user_ids), len(item_ids)).
      Значение ячейки — наибольший вес среди событий пары: вес события берётся из weights по столбцу
      event (по умолчанию DEFAULT_WEIGHTS: просмотр 1, покупка 3). Повторные события не суммируются.
      События, типа которых нет в weights, игнорируются; явных нулей в матрице быть не должно.
    """
    raise NotImplementedError


def precision_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Доля релевантных товаров среди первых k рекомендаций одного пользователя.

    recommended — товары в порядке убывания оценки, без повторов; relevant — множество релевантных товаров.
    Знаменатель всегда равен k, даже если рекомендаций меньше.
    """
    raise NotImplementedError


def recall_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Доля релевантных товаров пользователя, попавших в первые k рекомендаций.

    Если relevant пусто, возвращает 0.0.
    """
    raise NotImplementedError


def ndcg_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """NDCG@k с бинарной релевантностью для одного пользователя.

    DCG = сумма 1 / log2(позиция + 1) по позициям (с единицы) релевантных товаров среди первых k.
    IDCG — DCG идеального списка, в котором первые min(k, len(relevant)) позиций релевантны.
    Если relevant пусто, возвращает 0.0.
    """
    raise NotImplementedError


def catalog_coverage(recommendations: Sequence[Sequence[int]], n_items: int) -> float:
    """Доля каталога, попавшая в рекомендации хотя бы одному пользователю.

    recommendations — списки рекомендаций пользователей (или двумерный массив), n_items — размер каталога.
    """
    raise NotImplementedError


class PopularRecommender:
    """Базовая линия: самые популярные товары.

    Популярность товара — число пользователей с ненулевым значением в его столбце.
    """

    def fit(self, matrix: sp.csr_matrix) -> "PopularRecommender":
        """Запоминает матрицу и считает популярность товаров. Возвращает self."""
        raise NotImplementedError

    def recommend(self, user_index: int | None, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        """Возвращает массив из k индексов товаров по убыванию популярности.

        - при равной популярности выше товар с меньшим индексом;
        - если exclude_seen=True, товары с ненулевым значением в строке пользователя не рекомендуются;
        - user_index=None означает неизвестного (холодного) пользователя: исключать нечего.
        """
        raise NotImplementedError


class ItemKNNRecommender:
    """Item-based kNN с косинусным сходством.

    Сходство товаров — косинус между столбцами бинаризованной матрицы (ненулевое значение = 1).
    Сходство товара с самим собой равно нулю; для каждого товара хранятся только n_neighbors
    наиболее похожих. Оценка товара j для пользователя u — сумма по товарам i из его истории
    произведений matrix[u, i] · sim(i, j).
    """

    def __init__(self, n_neighbors: int = 50):
        self.n_neighbors = n_neighbors

    def fit(self, matrix: sp.csr_matrix) -> "ItemKNNRecommender":
        """Считает матрицу сходства товаров. Возвращает self.

        Требование к реализации: без циклов Python по парам товаров или пользователей.
        """
        raise NotImplementedError

    def recommend(self, user_index: int | None, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        """Возвращает массив из k индексов товаров по убыванию оценки, без повторов.

        - если exclude_seen=True, товары из истории пользователя не рекомендуются;
        - при равных оценках (в том числе нулевых) выше более популярный товар, как в PopularRecommender:
          если товаров с положительной оценкой меньше k, список дополняется популярными;
        - user_index=None или пользователь с пустой историей — рекомендации PopularRecommender.
        """
        raise NotImplementedError


class ALSRecommender:
    """ALS для неявного отклика (Hu, Koren, Volinsky).

    Предпочтение ячейки — 1, если значение матрицы ненулевое, иначе 0; уверенность c = 1 + alpha · r,
    где r — значение матрицы. Минимизируется сумма по всем ячейкам c · (предпочтение − x_u · y_i)²
    плюс reg · (сумма квадратов всех факторов).
    """

    def __init__(self, factors: int = 8, reg: float = 100.0, alpha: float = 10.0, n_iter: int = 15, seed: int = 0):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, matrix: sp.csr_matrix) -> "ALSRecommender":
        """Обучает факторы чередованием: n_iter раз пересчитывает факторы пользователей, затем товаров.

        После обучения доступны user_factors_ размера (число пользователей, factors) и item_factors_
        размера (число товаров, factors). Одинаковый seed даёт одинаковый результат. Возвращает self.
        Требование к реализации: без циклов по парам «пользователь–товар» и без перебора всех товаров
        для каждого пользователя.
        """
        raise NotImplementedError

    def recommend(self, user_index: int | None, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        """Возвращает массив из k индексов товаров по убыванию оценки x_u · y_i, без повторов.

        - если exclude_seen=True, товары из истории пользователя не рекомендуются;
        - user_index=None или пользователь с пустой историей — рекомендации PopularRecommender.
        """
        raise NotImplementedError


def recommend_ids(model, user_id: int, user_ids: np.ndarray, item_ids: np.ndarray, k: int = 10) -> list[int]:
    """Рекомендации в идентификаторах магазина — то, что отдаётся сайту.

    model — обученная модель с методом recommend; user_ids и item_ids — отображения из
    build_interaction_matrix. Возвращает список из k идентификаторов товаров (int) в порядке рекомендаций,
    уже виденные товары исключены. Если user_id нет в user_ids (новый пользователь), возвращаются
    рекомендации модели для холодного пользователя (user_index=None).
    """
    raise NotImplementedError
