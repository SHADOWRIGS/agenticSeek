import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))
sys.path.insert(0, str(PRACTICE_DIR.parent))

from shop_interactions import make_shop  # noqa: E402

from recsys import (  # noqa: E402
    ALSRecommender,
    ItemKNNRecommender,
    PopularRecommender,
    build_interaction_matrix,
    catalog_coverage,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recommend_ids,
    time_split,
)

SPLIT_DATE = pd.Timestamp("2025-06-01")
PERSONAL = {"item_knn": ItemKNNRecommender, "als": ALSRecommender}
_FIT_CACHE: dict = {}


def _try_fit(name: str, matrix: sp.csr_matrix, dataset: str):
    """Обучает модель один раз на набор данных. Возвращает (модель, секунды) или пойманную ошибку."""
    key = (name, dataset)
    if key not in _FIT_CACHE:
        try:
            start = time.perf_counter()
            model = PERSONAL[name]().fit(matrix)
            _FIT_CACHE[key] = (model, time.perf_counter() - start)
        except NotImplementedError as error:
            _FIT_CACHE[key] = error
    return _FIT_CACHE[key]


def fit_or_skip(name: str, matrix: sp.csr_matrix, dataset: str):
    """Обязательна одна персональная модель из двух: тесты нереализованной пропускаются."""
    result = _try_fit(name, matrix, dataset)
    if isinstance(result, NotImplementedError):
        other = next(other for other in PERSONAL if other != name)
        if isinstance(_try_fit(other, matrix, dataset), NotImplementedError):
            raise result
        pytest.skip(f"{PERSONAL[name].__name__} не реализован: выбрана другая модель")
    return result


def recommend_all(model, users: np.ndarray, k: int = 10) -> np.ndarray:
    return np.array([model.recommend(int(u), k) for u in users])


def mean_metric(metric, recs: np.ndarray, test: sp.csr_matrix, users: np.ndarray, k: int = 10) -> float:
    values = [
        metric(row, test.indices[test.indptr[u] : test.indptr[u + 1]], k) for row, u in zip(recs, users, strict=True)
    ]
    return float(np.mean(values))


# ---------- маленькие данные с известным ответом ----------


@pytest.fixture(scope="module")
def toy_events() -> pd.DataFrame:
    rows = [
        (20, 7, "view", "2025-01-01 10:00"),
        (10, 5, "view", "2025-01-02 09:00"),
        (10, 5, "purchase", "2025-01-02 09:05"),
        (10, 7, "view", "2025-01-03 00:00"),
        (10, 7, "view", "2025-01-04 12:00"),
        (30, 9, "purchase", "2025-01-05 08:00"),
        (20, 9, "cart", "2025-01-06 08:00"),
    ]
    events = pd.DataFrame(rows, columns=["user_id", "item_id", "event", "timestamp"])
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    return events


@pytest.fixture(scope="module")
def small_matrix() -> sp.csr_matrix:
    # популярность товаров: 3, 2, 2, 1, 0, 0; последний пользователь без истории
    dense = [
        [1, 1, 0, 0, 0, 0],
        [1, 0, 3, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    return sp.csr_matrix(np.array(dense, dtype=float))


@pytest.fixture(scope="module")
def block_matrix() -> sp.csr_matrix:
    """Два кластера по 20 пользователей и 6 товаров; каждый видел 5 из 6 товаров своего кластера.

    Пользователь u не видел товар с номером u % 6 внутри своего кластера. Строка 40 — пустая история.
    """
    dense = np.zeros((41, 12))
    for u in range(40):
        cluster = u // 20
        dense[u, cluster * 6 : cluster * 6 + 6] = 1
        dense[u, cluster * 6 + u % 6] = 0
    return sp.csr_matrix(dense)


class TestTimeSplit:
    def test_boundary_event_goes_to_test(self, toy_events):
        train, test = time_split(toy_events, pd.Timestamp("2025-01-03"))
        assert len(train) == 3
        assert len(test) == 4
        assert train["timestamp"].max() < pd.Timestamp("2025-01-03")
        assert test["timestamp"].min() == pd.Timestamp("2025-01-03")

    def test_columns_kept_and_source_untouched(self, toy_events):
        before = toy_events.copy()
        train, test = time_split(toy_events, pd.Timestamp("2025-01-03"))
        assert list(train.columns) == list(toy_events.columns)
        assert list(test.columns) == list(toy_events.columns)
        pd.testing.assert_frame_equal(toy_events, before)

    def test_all_events_before_or_after(self, toy_events):
        train, test = time_split(toy_events, pd.Timestamp("2024-01-01"))
        assert len(train) == 0 and len(test) == len(toy_events)
        train, test = time_split(toy_events, pd.Timestamp("2026-01-01"))
        assert len(train) == len(toy_events) and len(test) == 0


class TestBuildInteractionMatrix:
    def test_ids_are_sorted_and_define_shape(self, toy_events):
        matrix, user_ids, item_ids = build_interaction_matrix(toy_events)
        assert list(user_ids) == [10, 20, 30]
        assert list(item_ids) == [5, 7, 9]
        assert matrix.shape == (3, 3)

    def test_cell_is_max_weight_of_pair_events(self, toy_events):
        matrix, _, _ = build_interaction_matrix(toy_events)
        expected = [[3, 1, 0], [0, 1, 0], [0, 0, 3]]
        assert matrix.toarray().tolist() == expected

    def test_csr_float_without_explicit_zeros(self, toy_events):
        matrix, _, _ = build_interaction_matrix(toy_events)
        assert isinstance(matrix, sp.csr_matrix)
        assert matrix.dtype.kind == "f"
        assert matrix.nnz == 4

    def test_custom_weights(self, toy_events):
        matrix, user_ids, _ = build_interaction_matrix(toy_events, weights={"purchase": 1.0})
        assert list(user_ids) == [10, 20, 30]
        assert matrix.toarray().tolist() == [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
        assert matrix.nnz == 2

    def test_given_ids_fix_indexing_and_drop_unknown(self, toy_events):
        matrix, user_ids, item_ids = build_interaction_matrix(
            toy_events, user_ids=np.array([10, 30, 99]), item_ids=np.array([9, 5])
        )
        assert list(user_ids) == [10, 30, 99]
        assert list(item_ids) == [9, 5]
        assert matrix.toarray().tolist() == [[0, 3], [3, 0], [0, 0]]


class TestMetrics:
    def test_manual_example(self):
        recommended, relevant = [0, 1, 2, 3, 4], {1, 4, 5}
        assert precision_at_k(recommended, relevant, 5) == pytest.approx(0.4)
        assert recall_at_k(recommended, relevant, 5) == pytest.approx(2 / 3)
        dcg = 1 / np.log2(3) + 1 / np.log2(6)
        idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)
        assert ndcg_at_k(recommended, relevant, 5) == pytest.approx(dcg / idcg)
        assert ndcg_at_k(recommended, relevant, 5) == pytest.approx(0.4776, abs=1e-4)

    def test_perfect_list(self):
        assert precision_at_k([3, 1, 2], {1, 2, 3}, 3) == pytest.approx(1.0)
        assert recall_at_k([3, 1, 2], {1, 2, 3}, 3) == pytest.approx(1.0)
        assert ndcg_at_k([3, 1, 2], {1, 2, 3}, 3) == pytest.approx(1.0)

    def test_no_hits(self):
        assert precision_at_k([7, 8], {1}, 2) == 0.0
        assert recall_at_k([7, 8], {1}, 2) == 0.0
        assert ndcg_at_k([7, 8], {1}, 2) == 0.0

    def test_empty_relevant_gives_zero(self):
        assert precision_at_k([1, 2], set(), 2) == 0.0
        assert recall_at_k([1, 2], set(), 2) == 0.0
        assert ndcg_at_k([1, 2], set(), 2) == 0.0

    def test_only_first_k_positions_count(self):
        assert precision_at_k([9, 1, 2], {1, 2}, 1) == 0.0
        assert recall_at_k([9, 1, 2], {1, 2}, 1) == 0.0
        assert precision_at_k([9, 1, 2], {1, 2}, 2) == pytest.approx(0.5)
        assert recall_at_k([9, 1, 2], {1, 2}, 2) == pytest.approx(0.5)

    def test_short_list_precision_divides_by_k(self):
        assert precision_at_k([1], {1}, 5) == pytest.approx(0.2)
        assert recall_at_k([1], {1}, 5) == pytest.approx(1.0)
        assert ndcg_at_k([1], {1}, 5) == pytest.approx(1.0)

    def test_ndcg_ideal_uses_at_most_k_positions(self):
        assert ndcg_at_k([1, 2], {1, 2, 3, 4}, 2) == pytest.approx(1.0)

    def test_ndcg_depends_on_position(self):
        assert ndcg_at_k([1, 8, 9], {1}, 3) == pytest.approx(1.0)
        assert ndcg_at_k([8, 1, 9], {1}, 3) == pytest.approx(1 / np.log2(3))
        assert ndcg_at_k([8, 9, 1], {1}, 3) == pytest.approx(0.5)

    def test_accepts_arrays_and_lists(self):
        assert precision_at_k(np.array([0, 1, 2]), np.array([1, 2]), 3) == pytest.approx(2 / 3)
        assert recall_at_k(np.array([0, 1, 2]), [1, 2, 5, 6], 3) == pytest.approx(0.5)

    def test_catalog_coverage(self):
        assert catalog_coverage([[0, 1], [1, 2]], 10) == pytest.approx(0.3)
        assert catalog_coverage(np.array([[4, 4], [4, 4]]), 8) == pytest.approx(0.125)


class TestPopularRecommender:
    def test_fit_returns_self(self, small_matrix):
        model = PopularRecommender()
        assert model.fit(small_matrix) is model

    def test_order_by_number_of_users_ties_by_index(self, small_matrix):
        model = PopularRecommender().fit(small_matrix)
        assert list(model.recommend(None, k=6)) == [0, 1, 2, 3, 4, 5]

    def test_excludes_seen_items(self, small_matrix):
        model = PopularRecommender().fit(small_matrix)
        assert list(model.recommend(0, k=3)) == [2, 3, 4]
        assert list(model.recommend(2, k=2)) == [2, 4]

    def test_can_keep_seen_items(self, small_matrix):
        model = PopularRecommender().fit(small_matrix)
        assert list(model.recommend(0, k=3, exclude_seen=False)) == [0, 1, 2]

    def test_user_without_history(self, small_matrix):
        model = PopularRecommender().fit(small_matrix)
        assert list(model.recommend(4, k=3)) == [0, 1, 2]

    def test_returns_integer_array_of_length_k(self, small_matrix):
        recs = PopularRecommender().fit(small_matrix).recommend(1, k=2)
        assert isinstance(recs, np.ndarray)
        assert recs.dtype.kind == "i"
        assert len(recs) == 2


@pytest.fixture(scope="module", params=list(PERSONAL))
def toy_personal(request, block_matrix):
    return fit_or_skip(request.param, block_matrix, "block")[0]


class TestPersonalModelOnBlocks:
    def test_fit_returns_self(self, toy_personal, block_matrix):
        assert toy_personal.fit(block_matrix) is toy_personal

    @pytest.mark.parametrize("user", [0, 7, 19, 20, 33])
    def test_first_recommendation_is_unseen_item_of_own_cluster(self, toy_personal, user):
        expected = (user // 20) * 6 + user % 6
        assert toy_personal.recommend(user, k=1)[0] == expected

    def test_length_no_duplicates_no_seen(self, toy_personal, block_matrix):
        for user in range(40):
            recs = toy_personal.recommend(user, k=5)
            assert len(recs) == 5
            assert len(set(recs.tolist())) == 5
            assert block_matrix[user, recs].nnz == 0

    def test_unknown_user_gets_popular(self, toy_personal, block_matrix):
        popular = PopularRecommender().fit(block_matrix)
        assert list(toy_personal.recommend(None, k=4)) == list(popular.recommend(None, k=4))

    def test_user_without_history_gets_popular(self, toy_personal, block_matrix):
        popular = PopularRecommender().fit(block_matrix)
        assert list(toy_personal.recommend(40, k=4)) == list(popular.recommend(40, k=4))


class TestItemKNNDetails:
    def test_cosine_scores_on_manual_example(self):
        # история пользователя 3 — товар 0; cos(0, 1) = 2/√6, cos(0, 2) = 1/√6, cos(0, 3) = 0
        matrix = sp.csr_matrix(np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 0]], dtype=float))
        model = fit_or_skip("item_knn", matrix, "manual")[0]
        assert list(model.recommend(3, k=3)) == [1, 2, 3]

    def test_zero_scores_are_filled_by_popularity(self, block_matrix):
        model = fit_or_skip("item_knn", block_matrix, "block")[0]
        popular = PopularRecommender().fit(block_matrix)
        recs = model.recommend(0, k=4)
        other_cluster = [int(i) for i in popular.recommend(0, k=7) if i >= 6]
        assert list(recs[1:]) == other_cluster[:3]


class TestALSDetails:
    def test_factor_shapes(self, block_matrix):
        model = fit_or_skip("als", block_matrix, "block")[0]
        assert model.user_factors_.shape == (41, model.factors)
        assert model.item_factors_.shape == (12, model.factors)

    def test_same_seed_same_result(self, block_matrix):
        fit_or_skip("als", block_matrix, "block")
        first = ALSRecommender(factors=4, n_iter=3, seed=1).fit(block_matrix)
        second = ALSRecommender(factors=4, n_iter=3, seed=1).fit(block_matrix)
        assert np.allclose(first.item_factors_, second.item_factors_)


# ---------- данные магазина ----------


@pytest.fixture(scope="module")
def shop():
    return make_shop()


@pytest.fixture(scope="module")
def split(shop):
    return time_split(shop.events, SPLIT_DATE)


@pytest.fixture(scope="module")
def train(split):
    return build_interaction_matrix(split[0])


@pytest.fixture(scope="module")
def test_matrix(split, train) -> sp.csr_matrix:
    # цель — покупки июня; пользователи и товары, которых нет в обучении, не учитываются
    _, user_ids, item_ids = train
    return build_interaction_matrix(split[1], weights={"purchase": 1.0}, user_ids=user_ids, item_ids=item_ids)[0]


@pytest.fixture(scope="module")
def eval_users(test_matrix) -> np.ndarray:
    return np.flatnonzero(test_matrix.getnnz(axis=1) > 0)


@pytest.fixture(scope="module")
def popular_recs(train, eval_users) -> np.ndarray:
    return recommend_all(PopularRecommender().fit(train[0]), eval_users)


@pytest.fixture(scope="module", params=list(PERSONAL))
def personal(request, train):
    return fit_or_skip(request.param, train[0], "shop")


@pytest.fixture(scope="module")
def personal_recs(personal, eval_users):
    start = time.perf_counter()
    recs = recommend_all(personal[0], eval_users)
    return recs, time.perf_counter() - start


class TestOnShopData:
    def test_split_sizes(self, shop, split):
        train_events, test_events = split
        assert len(train_events) + len(test_events) == len(shop.events)
        assert train_events["timestamp"].max() < SPLIT_DATE <= test_events["timestamp"].min()

    def test_matrix_matches_events(self, split, train):
        matrix, user_ids, item_ids = train
        pairs = split[0].drop_duplicates(["user_id", "item_id"])
        assert matrix.shape == (split[0]["user_id"].nunique(), split[0]["item_id"].nunique())
        assert matrix.nnz == len(pairs)
        assert set(np.unique(matrix.data)) == {1.0, 3.0}
        bought = split[0][split[0]["event"] == "purchase"].iloc[0]
        row = np.flatnonzero(user_ids == bought["user_id"])[0]
        col = np.flatnonzero(item_ids == bought["item_id"])[0]
        assert matrix[row, col] == 3.0

    def test_popular_baseline_quality(self, popular_recs, test_matrix, eval_users):
        recall = mean_metric(recall_at_k, popular_recs, test_matrix, eval_users)
        assert 0.06 < recall < 0.09

    def test_personal_model_beats_popular(self, personal_recs, popular_recs, test_matrix, eval_users):
        recs = personal_recs[0]
        for metric in (recall_at_k, precision_at_k, ndcg_at_k):
            personal_value = mean_metric(metric, recs, test_matrix, eval_users)
            popular_value = mean_metric(metric, popular_recs, test_matrix, eval_users)
            assert personal_value > 1.8 * popular_value, metric.__name__

    def test_personal_recall_level(self, personal_recs, test_matrix, eval_users):
        assert mean_metric(recall_at_k, personal_recs[0], test_matrix, eval_users) > 0.15

    def test_recommendations_are_valid(self, personal_recs, train, eval_users):
        recs, matrix = personal_recs[0], train[0]
        assert recs.shape == (len(eval_users), 10)
        assert recs.min() >= 0 and recs.max() < matrix.shape[1]
        assert all(len(set(row.tolist())) == 10 for row in recs)
        seen = matrix[np.repeat(eval_users, 10), recs.ravel()]
        assert np.count_nonzero(seen) == 0

    def test_personal_model_covers_more_of_catalog(self, personal_recs, popular_recs, train):
        n_items = train[0].shape[1]
        assert catalog_coverage(popular_recs, n_items) < 0.03
        assert catalog_coverage(personal_recs[0], n_items) > 0.08

    def test_speed(self, personal, personal_recs):
        # эталон: обучение — около секунды, рекомендации для всех пользователей — доли секунды
        assert personal[1] < 20, "обучение слишком медленное: нет ли циклов по парам?"
        assert personal_recs[1] < 20, "рекомендации слишком медленные"

    def test_recommend_ids_for_known_user(self, personal, split, train, eval_users):
        model, (_, user_ids, item_ids) = personal[0], train
        user_index = int(eval_users[0])
        user_id = int(user_ids[user_index])
        result = recommend_ids(model, user_id, user_ids, item_ids, k=10)
        assert isinstance(result, list) and all(isinstance(item, int) for item in result)
        assert result == [int(item_ids[j]) for j in model.recommend(user_index, 10)]
        history = split[0].loc[split[0]["user_id"] == user_id, "item_id"]
        assert not set(result) & set(history)

    def test_recommend_ids_for_new_user(self, personal, train):
        model, (matrix, user_ids, item_ids) = personal[0], train
        popular = PopularRecommender().fit(matrix)
        expected = [int(item_ids[j]) for j in popular.recommend(None, 5)]
        assert recommend_ids(model, -1, user_ids, item_ids, k=5) == expected
