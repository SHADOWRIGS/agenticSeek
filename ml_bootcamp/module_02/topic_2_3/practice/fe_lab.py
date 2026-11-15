"""Практика 2.3.4: стенд экспериментов с признаками.

Реализуйте функции этого модуля. Ожидаемое поведение описано в docstring каждой функции
и проверяется тестами: uv run pytest -q

Стенд не привязан к конкретным данным: он работает с любой таблицей pandas, поэтому одинаково
подходит и для демо-данных из make_demo_data.py, и для настоящего конкурса.
"""

from pathlib import Path

import pandas as pd

JOURNAL_COLUMNS = ["time", "name", "n_features", "features", "scores", "mean", "std", "note"]


def count_encode(train: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Частотное кодирование категориального столбца.

    Каждое значение заменяется числом строк train с таким же значением:

    - пропуск считается отдельной категорией и кодируется числом пропусков в train;
    - значение из test, которого нет в train, кодируется нулём;
    - результат — два Series типа int64 с индексами и именами исходных train и test;
    - входные данные не изменяются.
    """
    raise NotImplementedError


def out_of_fold_target_encode(
    train: pd.Series, test: pd.Series, target: pd.Series, n_splits: int = 5, smooth: float = 10.0, seed: int = 42
) -> tuple[pd.Series, pd.Series]:
    """Target encoding категориального столбца без утечки целевой переменной.

    train и test — столбец категорий в обучающей и тестовой выборках; target — числовая целевая переменная
    (0/1 или вещественная) с тем же индексом, что и train.

    Кодировка категории c, посчитанная по набору строк S:

        (сумма target по строкам S с категорией c + smooth * prior) / (число таких строк + smooth),

    где prior — среднее target по всем строкам S. Категория, которой в S нет, кодируется значением prior.
    Пропуск считается отдельной категорией.

    - train кодируется по схеме out-of-fold: строки делятся на фолды через
      KFold(n_splits=n_splits, shuffle=True, random_state=seed) в том порядке, в каком они идут в train;
      строки каждого фолда кодируются по статистике, посчитанной на остальных фолдах;
    - test кодируется по статистике, посчитанной на всём train;
    - результат — два Series типа float64 без пропусков, с индексами и именами исходных train и test;
    - входные данные не изменяются.
    """
    raise NotImplementedError


def evaluate_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    model=None,
    scoring: str = "roc_auc",
    n_splits: int = 5,
    seed: int = 42,
) -> pd.Series:
    """Оценивает набор признаков кросс-валидацией и возвращает метрику на каждом фолде.

    - df — таблица с признаками и целевой переменной, target — имя столбца целевой переменной;
    - feature_columns не может быть пустым: иначе ValueError;
    - все столбцы из feature_columns должны быть числовыми или булевыми: иначе TypeError, в сообщении
      которого есть имя первого неподходящего столбца. Категории кодируются явно, до оценки;
    - model — любая модель scikit-learn. Если None — HistGradientBoostingClassifier(random_state=seed).
      Переданный объект не обучается: на каждом фолде обучается его копия (sklearn.base.clone);
    - scoring — название метрики scikit-learn, у которой «больше — лучше»: "roc_auc", "accuracy",
      "neg_root_mean_squared_error" и так далее;
    - разбиение: для классификатора (sklearn.base.is_classifier) —
      StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), иначе KFold с теми же параметрами;
    - результат — Series длины n_splits с именем "score"; индекс 0..n_splits-1 с именем "fold";
    - df не изменяется; повторный вызов с теми же аргументами даёт тот же результат.
    """
    raise NotImplementedError


def compare_feature_sets(
    df: pd.DataFrame, feature_sets: dict[str, list[str]], target: str, baseline: str, **kwargs
) -> pd.DataFrame:
    """Сравнивает наборы признаков на одних и тех же фолдах.

    feature_sets — словарь «название набора -> список столбцов»; baseline — название базового набора
    (если его нет в словаре — KeyError); kwargs передаются в evaluate_features без изменений.

    Результат — таблица: индекс с именем "feature_set" содержит названия наборов в порядке словаря, столбцы:

    - n_features: число признаков в наборе;
    - mean: средняя метрика по фолдам;
    - std: стандартное отклонение метрики по фолдам (ddof=1);
    - delta: mean набора минус mean базового набора (у базового набора — 0.0);
    - wins: в скольких фолдах метрика набора строго больше метрики базового набора на том же фолде
      (у базового набора — 0).
    """
    raise NotImplementedError


def log_experiment(path: Path, name: str, feature_columns: list[str], scores: pd.Series, note: str = "") -> dict:
    """Дописывает запись об эксперименте в журнал формата JSON Lines (одна строка файла — один JSON-объект).

    Запись — словарь с ключами из JOURNAL_COLUMNS:

    - "time": момент записи в UTC, строка в формате ISO 8601 (её понимает datetime.fromisoformat);
    - "name": название эксперимента;
    - "n_features": число признаков; "features": список их названий;
    - "scores": список метрик по фолдам (float); "mean": их среднее; "std": стандартное отклонение (ddof=1);
    - "note": комментарий.

    Файл и недостающие папки создаются при необходимости. Существующие записи не изменяются: новая запись
    добавляется в конец. Кодировка файла — UTF-8, кириллица записывается как есть (ensure_ascii=False).
    Функция возвращает записанный словарь.
    """
    raise NotImplementedError


def load_experiments(path: Path) -> pd.DataFrame:
    """Читает журнал экспериментов в таблицу.

    Одна строка — один эксперимент, столбцы — JOURNAL_COLUMNS в этом порядке. Строки отсортированы по
    убыванию mean, индекс — 0..n-1. Если файла нет — пустая таблица с теми же столбцами.
    """
    raise NotImplementedError
