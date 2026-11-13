import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

PRACTICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRACTICE_DIR))
sys.path.insert(0, str(PRACTICE_DIR.parent))

from customers_data import SNAPSHOT_DATE, make_transactions  # noqa: E402

from segmentation import (  # noqa: E402
    CLUSTER_FEATURES,
    FEATURES,
    LOG_FEATURES,
    SHARE_FEATURES,
    assign_segment,
    build_rfm_features,
    cluster_profiles,
    evaluate_k,
    preprocess,
    smooth_shares,
    stability_score,
)


def make_tiny_transactions() -> pd.DataFrame:
    # Строки намеренно перемешаны; заказы 7 и 8 сделаны не раньше даты среза 2025-01-01
    rows = [
        (5, 30, "2024-12-30", 400.0, False, "спорт", True),
        (1, 10, "2024-12-01", 1000.0, True, "книги", True),
        (7, 10, "2025-01-05", 9000.0, True, "дом", False),
        (3, 20, "2024-11-01", 500.0, True, "книги", False),
        (6, 30, "2024-12-31", 600.0, True, "спорт", True),
        (2, 10, "2024-12-21", 3000.0, False, "дом", True),
        (8, 40, "2025-01-01", 700.0, False, "дом", True),
        (4, 30, "2024-10-03", 200.0, False, "спорт", False),
    ]
    columns = ["order_id", "customer_id", "order_date", "amount", "promo", "category", "mobile"]
    tiny = pd.DataFrame(rows, columns=columns)
    tiny["order_date"] = pd.to_datetime(tiny["order_date"])
    return tiny


@pytest.fixture(scope="module")
def tiny_features() -> pd.DataFrame:
    return build_rfm_features(make_tiny_transactions(), "2025-01-01")


@pytest.fixture(scope="module")
def course_data() -> tuple[pd.DataFrame, pd.Series]:
    return make_transactions()


@pytest.fixture(scope="module")
def features(course_data) -> pd.DataFrame:
    return build_rfm_features(course_data[0], SNAPSHOT_DATE)


@pytest.fixture(scope="module")
def true_segment(course_data, features) -> pd.Series:
    return course_data[1].loc[features.index]


@pytest.fixture(scope="module")
def cluster_features(features) -> pd.DataFrame:
    return smooth_shares(features, SHARE_FEATURES, prior_strength=2.0)[CLUSTER_FEATURES]


@pytest.fixture(scope="module")
def prepared(cluster_features):
    return preprocess(cluster_features, LOG_FEATURES)


@pytest.fixture(scope="module")
def model(prepared) -> KMeans:
    return KMeans(n_clusters=6, n_init=10, random_state=0).fit(prepared[0])


@pytest.fixture(scope="module")
def blobs() -> np.ndarray:
    centers = [[0, 0], [10, 0], [0, 10], [10, 10]]
    return make_blobs(n_samples=400, centers=centers, cluster_std=1.0, random_state=0)[0]


@pytest.fixture(scope="module")
def blobs_metrics(blobs) -> pd.DataFrame:
    return evaluate_k(blobs, range(2, 8), seed=0)


class TestBuildRfmFeatures:
    def test_structure(self, tiny_features):
        assert list(tiny_features.columns) == FEATURES
        assert tiny_features.index.name == "customer_id"
        assert list(tiny_features.index) == [10, 20, 30]

    def test_values_on_manual_example(self, tiny_features):
        expected = pd.DataFrame(
            {
                "recency_days": [11, 61, 1],
                "frequency": [2, 1, 3],
                "monetary": [4000.0, 500.0, 1200.0],
                "avg_check": [2000.0, 500.0, 400.0],
                "promo_share": [0.5, 1.0, 1 / 3],
                "n_categories": [2, 1, 1],
                "mobile_share": [1.0, 0.0, 2 / 3],
                "account_age_days": [31, 61, 90],
            },
            index=pd.Index([10, 20, 30], name="customer_id"),
        )
        pd.testing.assert_frame_equal(tiny_features, expected, check_dtype=False)

    def test_orders_on_or_after_snapshot_are_ignored(self, tiny_features):
        assert 40 not in tiny_features.index
        assert tiny_features.loc[10, "frequency"] == 2
        assert tiny_features.loc[10, "monetary"] == pytest.approx(4000.0)

    def test_earlier_snapshot(self):
        early = build_rfm_features(make_tiny_transactions(), pd.Timestamp("2024-12-25"))
        assert list(early.index) == [10, 20, 30]
        assert early.loc[10, "recency_days"] == 4
        assert early.loc[30, "frequency"] == 1
        assert early.loc[30, "recency_days"] == 83
        assert early.loc[30, "promo_share"] == pytest.approx(0.0)

    def test_input_is_not_modified(self):
        tiny = make_tiny_transactions()
        before = tiny.copy()
        build_rfm_features(tiny, "2025-01-01")
        pd.testing.assert_frame_equal(tiny, before)

    def test_course_data(self, course_data, features):
        transactions = course_data[0]
        assert len(features) == transactions["customer_id"].nunique()
        assert features["frequency"].sum() == len(transactions)
        assert features["monetary"].sum() == pytest.approx(transactions["amount"].sum())
        assert features["recency_days"].min() >= 1
        assert (features["recency_days"] <= features["account_age_days"]).all()
        assert features.notna().all().all()


class TestSmoothShares:
    @pytest.fixture
    def two_customers(self) -> pd.DataFrame:
        return pd.DataFrame({"frequency": [1, 9], "promo_share": [1.0, 0.0], "avg_check": [100.0, 200.0]})

    def test_manual_example(self, two_customers):
        # Доля по всем заказам: 1 акционный заказ из 10, p0 = 0.1
        smoothed = smooth_shares(two_customers, ["promo_share"], prior_strength=2.0)
        assert smoothed["promo_share"].tolist() == pytest.approx([(1 + 0.2) / 3, (0 + 0.2) / 11])

    def test_zero_strength_keeps_shares(self, two_customers):
        smoothed = smooth_shares(two_customers, ["promo_share"], prior_strength=0.0)
        assert smoothed["promo_share"].tolist() == pytest.approx([1.0, 0.0])

    def test_other_columns_and_input_are_untouched(self, two_customers):
        before = two_customers.copy()
        smoothed = smooth_shares(two_customers, ["promo_share"], prior_strength=5.0)
        pd.testing.assert_frame_equal(two_customers, before)
        pd.testing.assert_frame_equal(smoothed.drop(columns="promo_share"), before.drop(columns="promo_share"))

    def test_rare_buyers_are_pulled_to_the_overall_share(self, features):
        smoothed = smooth_shares(features, SHARE_FEATURES, prior_strength=2.0)
        assert list(smoothed.columns) == list(features.columns)
        assert smoothed[SHARE_FEATURES].min().min() > 0
        assert smoothed[SHARE_FEATURES].max().max() < 1
        change = (smoothed["promo_share"] - features["promo_share"]).abs()
        assert change[features["frequency"] == 1].mean() > 5 * change[features["frequency"] >= 20].mean()


class TestPreprocess:
    def test_matrix_is_standardized(self, cluster_features, prepared):
        X = prepared[0]
        assert isinstance(X, np.ndarray)
        assert X.shape == cluster_features.shape
        assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
        assert np.allclose(X.std(axis=0), 1, atol=1e-8)

    def test_log_is_applied_to_listed_columns_only(self, cluster_features, prepared):
        expected = cluster_features.astype(float)
        expected[LOG_FEATURES] = np.log1p(expected[LOG_FEATURES])
        expected = (expected - expected.mean()) / expected.std(ddof=0)
        assert np.allclose(prepared[0], expected.to_numpy())

    def test_transformer_reproduces_training_matrix(self, cluster_features, prepared):
        X, transformer = prepared
        assert np.allclose(transformer.transform(cluster_features), X)
        assert np.allclose(transformer.transform(cluster_features.iloc[:5]), X[:5])

    def test_transformer_uses_training_statistics(self):
        train = pd.DataFrame({"a": [0.0, np.e - 1, np.e**2 - 1], "b": [1.0, 2.0, 3.0]})
        X, transformer = preprocess(train, ["a"])
        # log1p(a) = 0, 1, 2; среднее 1, отклонение sqrt(2/3)
        assert np.allclose(X[:, 0], np.array([-1.0, 0.0, 1.0]) / np.sqrt(2 / 3))
        new = pd.DataFrame({"a": [np.e**3 - 1], "b": [2.0]})
        assert np.allclose(transformer.transform(new), [[2.0 / np.sqrt(2 / 3), 0.0]])

    def test_input_is_not_modified(self, cluster_features):
        before = cluster_features.copy()
        preprocess(cluster_features, LOG_FEATURES)
        pd.testing.assert_frame_equal(cluster_features, before)


class TestEvaluateK:
    def test_structure(self, blobs_metrics):
        assert list(blobs_metrics.index) == [2, 3, 4, 5, 6, 7]
        assert blobs_metrics.index.name == "k"
        assert list(blobs_metrics.columns) == ["inertia", "silhouette", "calinski_harabasz", "davies_bouldin"]

    def test_values_match_sklearn(self, blobs, blobs_metrics):
        reference = KMeans(n_clusters=3, n_init=10, random_state=0).fit(blobs)
        row = blobs_metrics.loc[3]
        assert row["inertia"] == pytest.approx(reference.inertia_)
        assert row["silhouette"] == pytest.approx(silhouette_score(blobs, reference.labels_))
        assert row["calinski_harabasz"] == pytest.approx(calinski_harabasz_score(blobs, reference.labels_))
        assert row["davies_bouldin"] == pytest.approx(davies_bouldin_score(blobs, reference.labels_))

    def test_inertia_decreases(self, blobs_metrics):
        assert blobs_metrics["inertia"].is_monotonic_decreasing

    def test_true_number_of_clusters_wins_on_separated_blobs(self, blobs_metrics):
        assert blobs_metrics["silhouette"].idxmax() == 4
        assert blobs_metrics["calinski_harabasz"].idxmax() == 4
        assert blobs_metrics["davies_bouldin"].idxmin() == 4


class TestStabilityScore:
    def test_separated_blobs_are_stable(self, blobs):
        score = stability_score(blobs, k=4)
        assert isinstance(score, float)
        assert score > 0.98

    def test_wrong_k_is_less_stable(self, blobs):
        assert stability_score(blobs, k=7) < stability_score(blobs, k=4) - 0.1

    def test_noise_is_unstable(self):
        noise = np.random.default_rng(0).uniform(size=(1000, 7))
        score = stability_score(noise, k=6)
        assert -1 <= score < 0.6

    def test_same_seed_same_result(self, blobs):
        assert stability_score(blobs, k=5, n_runs=3, seed=1) == stability_score(blobs, k=5, n_runs=3, seed=1)


class TestClusterProfiles:
    @pytest.fixture
    def small(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"avg_check": [100.0, 300.0, 1000.0, 200.0, 5000.0], "frequency": [1, 3, 10, 2, 20]},
            index=[11, 12, 13, 14, 15],
        )

    def test_manual_example(self, small):
        profiles = cluster_profiles(small, np.array([1, 1, 0, 1, 0]))
        expected = pd.DataFrame(
            {"size": [2, 3], "share": [0.4, 0.6], "avg_check": [3000.0, 200.0], "frequency": [15.0, 2.0]},
            index=pd.Index([0, 1], name="cluster"),
        )
        pd.testing.assert_frame_equal(profiles, expected, check_dtype=False)

    def test_series_labels_are_matched_by_position(self, small):
        labels = pd.Series([1, 1, 0, 1, 0])  # индекс 0..4 не совпадает с индексом таблицы
        assert cluster_profiles(small, labels)["size"].tolist() == [2, 3]

    def test_course_data(self, features, model):
        profiles = cluster_profiles(features, model.labels_)
        assert list(profiles.columns) == ["size", "share", *FEATURES]
        assert profiles["size"].sum() == len(features)
        assert profiles["share"].sum() == pytest.approx(1.0)
        # Профили — в рублях и днях, а не в стандартизованных единицах
        assert profiles["avg_check"].min() > 500
        assert profiles["account_age_days"].max() > 365


class TestAssignSegment:
    def test_training_customers_keep_their_clusters(self, cluster_features, prepared, model):
        assigned = assign_segment(cluster_features, prepared[1], model)
        assert assigned.name == "segment"
        assert assigned.index.equals(cluster_features.index)
        assert (assigned.to_numpy() == model.labels_).all()

    def test_segment_names(self, cluster_features, prepared, model):
        names = {cluster: f"сегмент {cluster}" for cluster in range(6)}
        assigned = assign_segment(cluster_features.iloc[:50], prepared[1], model, segment_names=names)
        assert assigned.tolist() == [names[cluster] for cluster in model.labels_[:50]]

    def test_single_row(self, cluster_features, prepared, model):
        assigned = assign_segment(cluster_features.iloc[[7]], prepared[1], model)
        assert len(assigned) == 1
        assert assigned.iloc[0] == model.labels_[7]

    def test_new_customers(self, prepared, model):
        transactions, segment = make_transactions(n=2000, seed=7)
        new = smooth_shares(build_rfm_features(transactions, SNAPSHOT_DATE), SHARE_FEATURES)[CLUSTER_FEATURES]
        assigned = assign_segment(new, prepared[1], model)
        assert adjusted_rand_score(segment.loc[new.index], assigned) > 0.8


class TestPipelineOnCourseData:
    def test_hidden_segments_are_recovered(self, true_segment, model):
        assert adjusted_rand_score(true_segment, model.labels_) > 0.85

    def test_no_tiny_clusters(self, model):
        assert np.bincount(model.labels_).min() >= 0.03 * len(model.labels_)

    def test_six_clusters_are_stable(self, prepared):
        assert stability_score(prepared[0], k=6) > 0.95

    def test_internal_metrics_prefer_five_or_six(self, prepared):
        metrics = evaluate_k(prepared[0], range(3, 9))
        assert metrics["silhouette"].idxmax() in (5, 6)

    def test_wholesale_segment_is_visible_in_profiles(self, features, model):
        profiles = cluster_profiles(features, model.labels_)
        smallest = profiles["size"].idxmin()
        assert profiles.loc[smallest, "avg_check"] > 15_000
        assert profiles.loc[smallest, "frequency"] > 20
