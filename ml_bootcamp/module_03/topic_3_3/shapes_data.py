"""Двумерные наборы точек с известными кластерами для уроков темы 3.3.

Использование:  from shapes_data import make_shapes
"""

import numpy as np
from sklearn.datasets import make_blobs, make_moons


def make_shapes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Семь наборов по 1500 точек: название -> (координаты, истинные метки). Шум помечен -1.

    - blobs: три круглых облака одного размера — случай, для которого придуман k-means;
    - moons: два вложенных полумесяца;
    - aniso: три вытянутых наклонных облака;
    - variance: три облака с разным разбросом (среднее втрое шире крайних);
    - sizes: одно большое облако (1350 точек) и два маленьких плотных (по 75);
    - density: два плотных облака рядом и одно разреженное в стороне;
    - noise: набор blobs, в котором десятая часть точек заменена равномерным шумом.
    """
    shapes = {}
    shapes["blobs"] = make_blobs(n_samples=1500, centers=3, cluster_std=1.0, random_state=3)
    shapes["moons"] = make_moons(n_samples=1500, noise=0.06, random_state=0)
    points, labels = make_blobs(n_samples=1500, centers=3, cluster_std=1.0, random_state=170)
    shapes["aniso"] = (points @ np.array([[0.6, -0.6], [-0.4, 0.8]]), labels)
    shapes["variance"] = make_blobs(
        n_samples=1500, centers=[[-5, 0], [0, 0], [5, 0]], cluster_std=[0.6, 2.2, 0.6], random_state=0
    )
    shapes["sizes"] = make_blobs(
        n_samples=[1350, 75, 75], centers=[[0, 0], [6, 0], [0, 6]], cluster_std=[1.5, 0.4, 0.4], random_state=0
    )
    shapes["density"] = make_blobs(
        n_samples=[500, 500, 500], centers=[[0, 0], [1.6, 0], [8, 5]], cluster_std=[0.25, 0.25, 2.0], random_state=0
    )
    points, labels = make_blobs(n_samples=1350, centers=3, cluster_std=1.0, random_state=3)
    noise = np.random.default_rng(0).uniform(points.min(axis=0) - 3, points.max(axis=0) + 3, size=(150, 2))
    shapes["noise"] = (np.vstack([points, noise]), np.concatenate([labels, np.full(150, -1)]))
    return shapes
