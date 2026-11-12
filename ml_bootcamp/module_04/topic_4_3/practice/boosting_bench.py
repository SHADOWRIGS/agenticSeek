"""Практика 4.3.4: стенд для честного сравнения библиотек градиентного бустинга.

Реализуйте функции, в которых стоит raise NotImplementedError. Ожидаемое поведение описано
в docstring и проверяется тестами: uv run pytest -q

Сравниваются четыре библиотеки: XGBoost, LightGBM, CatBoost и HistGradientBoosting из scikit-learn.
Задача (task) — "regression" или "classification" (бинарная и многоклассовая; метки классов — целые
числа 0..K-1). Метрика качества — функция потерь, чем меньше, тем лучше: RMSE для регрессии, log loss
для классификации. Входные данные изменять нельзя.
"""

import pandas as pd

LIBRARIES = ("xgboost", "lightgbm", "catboost", "sklearn")
TASKS = ("regression", "classification")
# Строка, которой заменяются пропуски в категориальных признаках при kind="string"
MISSING = "__missing__"
RESULT_COLUMNS = ["score", "valid_score", "n_trees", "fit_time", "predict_time"]


def make_models(
    task: str,
    random_state: int = 0,
    max_trees: int = 1000,
    learning_rate: float = 0.1,
    n_jobs: int = 1,
    overrides: dict[str, dict] | None = None,
) -> dict:
    """Необученные модели четырёх библиотек с сопоставимыми параметрами.

    Возвращает словарь с ключами LIBRARIES в этом порядке:
        "xgboost"  — XGBRegressor / XGBClassifier;
        "lightgbm" — LGBMRegressor / LGBMClassifier;
        "catboost" — CatBoostRegressor / CatBoostClassifier;
        "sklearn"  — HistGradientBoostingRegressor / HistGradientBoostingClassifier.

    Общие параметры задаются у всех одинаково, под именем, принятым в библиотеке:
        max_trees     — верхняя граница числа итераций бустинга: n_estimators (xgboost, lightgbm),
                        iterations (catboost), max_iter (sklearn);
        learning_rate — шаг обучения;
        random_state  — random_state (xgboost, lightgbm, sklearn), random_seed (catboost);
        n_jobs        — число потоков: n_jobs (xgboost, lightgbm), thread_count (catboost);
                        у sklearn такого параметра нет.
    Остальные параметры — значения по умолчанию каждой библиотеки, кроме служебных:
        xgboost  — enable_categorical=True;
        lightgbm — verbose=-1;
        catboost — verbose=0, allow_writing_files=False (библиотека не должна создавать catboost_info);
        sklearn  — early_stopping=False (обычный fit строит ровно max_trees итераций).

    overrides — дополнительные параметры по библиотекам, например {"xgboost": {"max_depth": 3}}.
    Они передаются в конструктор и имеют приоритет над общими параметрами.

    ValueError — при неизвестной задаче и при неизвестном ключе в overrides.
    Каждый вызов возвращает новые объекты; модели не обучены.
    """
    raise NotImplementedError


def split_train_valid_test(
    X,
    y,
    valid_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 0,
    stratify: bool = False,
    time=None,
) -> tuple:
    """Разбиение на обучение, валидацию и тест.

    Возвращает (X_train, X_valid, X_test, y_train, y_valid, y_test). Размеры частей:
    round(n * test_size) объектов в тесте, round(n * valid_size) в валидации, остальные в обучении.
    Части не пересекаются и вместе дают все объекты. Типы сохраняются: DataFrame и Series остаются
    DataFrame и Series с исходными метками индекса, массивы NumPy — массивами.

    time is None — случайное разбиение, воспроизводимое при одинаковом random_state.
        При stratify=True доли классов y в каждой части такие же, как во всей выборке.
    time задан (массив длины n) — разбиение по времени, без случайности: самые поздние объекты
        попадают в тест, предшествующие им — в валидацию, самые ранние — в обучение. Внутри каждой
        части объекты упорядочены по времени. random_state и stratify в этом режиме не используются.
    """
    raise NotImplementedError


def prepare_categoricals(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    cat_features: list[str],
    kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Подготовка категориальных столбцов под конкретную библиотеку.

    Возвращает копии трёх таблиц; столбцы, не входящие в cat_features, не меняются. Список известных
    категорий каждого столбца — отсортированные значения из train без пропусков; valid и test на него
    не влияют.

    kind="category" — для xgboost, lightgbm и sklearn: тип pandas category с известными категориями
        (одинаковый во всех трёх таблицах). Пропуски и значения, которых не было в train, становятся
        пропусками. Предупреждений pandas возникать не должно.
    kind="string" — для catboost: все значения — строки (str); пропуски заменяются строкой MISSING;
        значения, которых не было в train, остаются как есть.
    kind="ordinal" — целочисленный номер категории в списке известных (от 0); пропуски и значения,
        которых не было в train, получают -1.

    ValueError — при неизвестном kind.
    """
    raise NotImplementedError


def fit_with_early_stopping(model, X_train, y_train, X_valid, y_valid, rounds: int = 50, cat_features=None) -> tuple:
    """Обучение с ранней остановкой по валидационной выборке — единый интерфейс для четырёх библиотек.

    Возвращает (обученная модель, n_trees). Переданная модель не изменяется и не обучается: обучается
    её копия (sklearn.base.clone). Верхняя граница числа итераций — та, что задана в самой модели.
    Обучение прекращается, если функция потерь на (X_valid, y_valid) не улучшается rounds итераций
    подряд. Никакие другие данные для остановки не используются.

    n_trees — число итераций бустинга, которые модель использует при предсказании:
        xgboost  — best_iteration + 1 (остановка задаётся параметром early_stopping_rounds модели);
        lightgbm — best_iteration_ (остановка — callback lightgbm.early_stopping, без вывода на экран);
        catboost — tree_count_ (остановка — early_stopping_rounds в fit; модель обрезается до лучшей
                   итерации);
        sklearn  — n_iter_ (early_stopping=True, n_iter_no_change=rounds, валидационные данные —
                   аргументы X_val и y_val метода fit; отката к лучшей итерации в scikit-learn нет).
    Следствие, которое проверяют тесты: модель, обученная без остановки ровно на n_trees итераций,
    даёт те же предсказания (исключение — классификация в catboost: там результат сравнивается
    с первыми n_trees деревьями модели, обученной без остановки).

    cat_features — имена категориальных столбцов; передаются только в catboost (метод fit),
    остальные библиотеки определяют их по типу category.

    TypeError — если модель не относится к четырём поддерживаемым классам.
    Предупреждений об устаревших аргументах библиотек возникать не должно.
    """
    raise NotImplementedError


def evaluate(model, X, y, task: str) -> float:
    """Качество обученной модели на данных (X, y): чем меньше, тем лучше.

    "regression" — RMSE предсказаний model.predict(X).
    "classification" — log loss вероятностей model.predict_proba(X); порядок столбцов задаёт
        model.classes_. В y могут встречаться не все классы — метрика всё равно должна считаться.
    ValueError — при неизвестной задаче.
    """
    raise NotImplementedError


def benchmark(models: dict, data: tuple, task: str, rounds: int = 50, cat_features=None) -> pd.DataFrame:
    """Сравнение моделей на одном наборе данных.

    models — словарь «название → необученная модель» (например, результат make_models или его часть;
        названия произвольные);
    data — (X_train, X_valid, X_test, y_train, y_valid, y_test), как возвращает split_train_valid_test;
    cat_features — имена категориальных столбцов или None. Если заданы, таблицы для каждой модели
        готовятся функцией prepare_categoricals: kind="string" для catboost (и cat_features передаются
        в fit_with_early_stopping), kind="category" для остальных.

    Каждая модель обучается функцией fit_with_early_stopping на обучающей части с остановкой
    по валидационной. Тестовая часть используется только для итоговой оценки: от неё не должны зависеть
    ни число деревьев, ни качество на валидации.

    Возвращает DataFrame: индекс с именем "model" — названия моделей в порядке словаря; столбцы
        score        — evaluate на тесте;
        valid_score  — evaluate на валидации;
        n_trees      — число итераций из fit_with_early_stopping (целое);
        fit_time     — время обучения, секунды;
        predict_time — время предсказания для тестовой части, секунды.
    При одинаковых моделях и данных score, valid_score и n_trees воспроизводятся точно.
    """
    raise NotImplementedError


def summarize(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Сводка по нескольким наборам данных.

    results — словарь «название набора → таблица benchmark»; во всех таблицах должны быть одни и те же
    модели, иначе ValueError.

    Возвращает DataFrame с индексом "model" и столбцами:
        mean_rank       — средний ранг модели по наборам. Ранг считается внутри набора по score:
                          1 — лучшая (наименьший score); при равенстве — средний ранг;
        wins            — число наборов, на которых score модели равен лучшему;
        mean_ratio      — среднее по наборам отношение score модели к лучшему score набора
                          (1.0 — модель везде лучшая);
        total_fit_time  — суммарное время обучения по всем наборам.
    Строки отсортированы по mean_rank по возрастанию; при равенстве — по названию модели.
    """
    raise NotImplementedError
