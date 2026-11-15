"""Сквозной ML-пайплайн прогноза дневных продаж: практика 7.1.4.

Пайплайн за дату run_date: выгрузить партиции -> проверить -> обучить -> оценить -> опубликовать модель,
если она достаточно хороша. Каждый этап — функция с явными входами, результат — файлы на диске.

Раскладка рабочего каталога workdir:

    data/raw/date=YYYY-MM-DD/sales.csv      партиция: продажи за один день
    runs/YYYY-MM-DD/model.joblib            артефакты запуска за дату
    runs/YYYY-MM-DD/metrics.json
    runs/YYYY-MM-DD/metadata.json
    registry/YYYY-MM-DD/model.joblib        опубликованные версии модели
    registry/YYYY-MM-DD/metrics.json
    registry/current.json                   указатель на текущую версию

Готовые части менять не нужно: константы, DataValidationError, DateFeatures, partition_path, window_days,
sales_source и main. Реализуйте остальное так, чтобы проходили тесты из tests/test_ml_pipeline.py.
"""

import argparse
from collections.abc import Callable
from datetime import date, timedelta
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

Source = Callable[[date], pd.DataFrame]

TARGET = "units"
KEYS = ["store_id", "product"]
REQUIRED_COLUMNS = ["date", "store_id", "product", "price", "promo", "units"]
OTHER = "__other__"
# Наибольшая допустимая доля пропусков целевой переменной в одной партиции
MAX_MISSING_SHARE = 0.05

DEFAULT_CONFIG = {
    "train_days": 56,  # длина обучающего окна, дней
    "valid_days": 7,  # длина проверочного окна, дней; его последний день — run_date
    "min_category_count": 20,  # порог редкой категории для RareCategoryGrouper
    "alpha": 1.0,  # регуляризация Ridge
    "max_mae_ratio": 0.8,  # порог публикации: MAE модели / MAE прогноза средним
}


class DataValidationError(ValueError):
    """Партиция не прошла проверку."""


class DateFeatures(TransformerMixin, BaseEstimator):
    """Готовый трансформер из урока 7.1.1: признаки из одного столбца дат."""

    def __init__(self, origin="2023-01-01"):
        self.origin = origin

    def fit(self, X, y=None):
        self._dates(X)
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        dates = self._dates(X)
        angle = 2 * np.pi * dates.dt.day_of_year / 365.25
        return pd.DataFrame(
            {
                "dow": dates.dt.day_of_week,
                "is_weekend": (dates.dt.day_of_week >= 5).astype(int),
                "doy_sin": np.sin(angle),
                "doy_cos": np.cos(angle),
                "days": (dates - pd.Timestamp(self.origin)).dt.days,
            },
            index=dates.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["dow", "is_weekend", "doy_sin", "doy_cos", "days"], dtype=object)

    @staticmethod
    def _dates(X) -> pd.Series:
        if not isinstance(X, pd.DataFrame) or X.shape[1] != 1:
            raise ValueError("DateFeatures ожидает таблицу из одного столбца с датами")
        return pd.to_datetime(X.iloc[:, 0])


def partition_path(data_dir: str | Path, day: date) -> Path:
    """Путь к партиции за день: data_dir/raw/date=YYYY-MM-DD/sales.csv."""
    return Path(data_dir) / "raw" / f"date={day:%Y-%m-%d}" / "sales.csv"


def window_days(run_date: date, config: dict) -> tuple[list[date], list[date]]:
    """Дни обучающего и проверочного окон по возрастанию.

    Проверочное окно — последние valid_days дней, включая run_date; обучающее — train_days дней перед ним.
    """
    valid_start = run_date - timedelta(days=config["valid_days"] - 1)
    train_start = valid_start - timedelta(days=config["train_days"])
    train = [train_start + timedelta(days=i) for i in range(config["train_days"])]
    valid = [valid_start + timedelta(days=i) for i in range(config["valid_days"])]
    return train, valid


class RareCategoryGrouper(TransformerMixin, BaseEstimator):
    """Объединяет редкие и незнакомые категории в одно значение OTHER.

    Вход fit и transform — pandas.DataFrame с категориальными столбцами; другой тип входа — TypeError.

    Контракт scikit-learn:
    - __init__ только сохраняет параметр min_count; проверка min_count >= 1 выполняется в fit (ValueError);
    - fit(X, y=None) возвращает self и создаёт атрибуты:
        categories_ — словарь «столбец -> отсортированный список значений, встретившихся в X не меньше
                      min_count раз» (пропуски не считаются значением);
        feature_names_in_ — массив названий столбцов X; n_features_in_ — их число;
    - до fit этих атрибутов нет, а transform и get_feature_names_out вызывают NotFittedError;
    - transform(X) возвращает новую таблицу с тем же индексом и теми же столбцами: значения из categories_
      сохраняются, все остальные — редкие, незнакомые и пропущенные — заменяются на OTHER. Столбцы X должны
      совпадать с feature_names_in_ (иначе ValueError);
    - transform не изменяет ни X, ни атрибуты объекта: статистики считаются только в fit;
    - get_feature_names_out() возвращает названия столбцов;
    - объект клонируется (sklearn.base.clone) и работает внутри Pipeline, ColumnTransformer и GridSearchCV.
    """

    def __init__(self, min_count=10):
        self.min_count = min_count

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def get_feature_names_out(self, input_features=None):
        raise NotImplementedError


def build_pipeline(config: dict) -> Pipeline:
    """Необученный пайплайн «признаки -> модель».

    Шаг "features" — ColumnTransformer с remainder="drop" и тремя элементами:
        "date": DateFeatures() для столбца ["date"];
        "cat":  Pipeline из шагов "rare" (RareCategoryGrouper с min_count=config["min_category_count"])
                и "onehot" (OneHotEncoder, handle_unknown="ignore", sparse_output=False) для KEYS;
        "num":  StandardScaler для ["price", "promo"].
    Шаг "model" — TransformedTargetRegressor: Ridge(alpha=config["alpha"]), целевая переменная
    преобразуется через np.log1p, прогноз — обратно через np.expm1.

    Пайплайн обучается на таблице со столбцами REQUIRED_COLUMNS; лишние столбцы (включая units)
    он отбрасывает сам. Прогноз для строки с незнакомым магазином или товаром не вызывает ошибку.
    """
    raise NotImplementedError


def extract(run_date: date, source: Source, data_dir: str | Path) -> Path:
    """Выгружает продажи за run_date в партицию и возвращает путь к ней.

    - данные — source(run_date); пустая таблица — ValueError, партиция при этом не создаётся;
    - файл — partition_path(data_dir, run_date), формат CSV без индекса;
    - идемпотентность: повторный вызов перезаписывает партицию целиком — тот же источник даёт тот же файл
      байт в байт, строки не дублируются;
    - запись атомарна: данные пишутся во временный файл в том же каталоге, затем он переименовывается.
      После вызова в каталоге партиции лежит ровно один файл.
    """
    raise NotImplementedError


def validate(path: str | Path, run_date: date) -> dict:
    """Проверяет партицию; возвращает {"rows": число строк, "missing_share": доля пропусков в units}.

    Нарушения (каждое — причина для DataValidationError; в сообщении перечисляются все найденные):
    - отсутствует столбец из REQUIRED_COLUMNS (дальнейшие проверки не выполняются);
    - в партиции нет строк;
    - есть строки с датой, отличной от run_date;
    - повторяется пара KEYS;
    - есть price <= 0 или units < 0;
    - доля пропусков в units больше MAX_MISSING_SHARE.
    Функция только читает файл.
    """
    raise NotImplementedError


def train(run_date: date, data_dir: str | Path, run_dir: str | Path, config: dict) -> Path:
    """Обучает build_pipeline(config) и сохраняет его в run_dir/model.joblib; возвращает путь к файлу.

    - данные — партиции дней обучающего окна (window_days); отсутствующая партиция — FileNotFoundError;
    - строки с пропуском в units в обучении не участвуют;
    - партиции проверочного окна не читаются: их содержимое не влияет на модель;
    - run_dir создаётся при необходимости; файл записывается через joblib.dump.
    """
    raise NotImplementedError


def evaluate(model_path: str | Path, run_date: date, data_dir: str | Path, run_dir: str | Path, config: dict) -> dict:
    """Оценивает модель на проверочном окне, записывает run_dir/metrics.json и возвращает тот же словарь.

    Строки с пропуском в units не учитываются. Ключи:
        "mae" — средняя абсолютная ошибка модели на проверочном окне;
        "baseline_mae" — ошибка прогноза одним числом: средним units обучающего окна;
        "mae_ratio" — mae / baseline_mae;
        "n_train", "n_valid" — число строк обучающего и проверочного окон с известным units.
    Значения — обычные float и int. Файл: json.dumps(metrics, indent=2, sort_keys=True).
    """
    raise NotImplementedError


def publish(run_dir: str | Path, registry_dir: str | Path, run_date: date, max_mae_ratio: float) -> bool:
    """Публикует модель запуска, если она не хуже порога; возвращает True, если модель опубликована.

    - решение принимается по run_dir/metrics.json: публикуем при mae_ratio <= max_mae_ratio;
    - при отказе реестр не изменяется (каталог registry_dir не создаётся);
    - при публикации model.joblib и metrics.json копируются в registry_dir/YYYY-MM-DD/; повторная
      публикация за ту же дату заменяет файлы версии, а не создаёт новую версию;
    - указатель registry_dir/current.json — {"run_date": "YYYY-MM-DD", "model": "YYYY-MM-DD/model.joblib"} —
      обновляется, только если в нём нет более поздней даты: пересчёт за прошлую дату не должен
      подменять текущую модель старой;
    - файлы реестра записываются атомарно; временных файлов после вызова не остаётся.
    """
    raise NotImplementedError


def run(run_date: date, workdir: str | Path, source: Source, config: dict | None = None) -> dict:
    """Выполняет пайплайн за run_date; возвращает {"run_dir": Path, "metrics": dict, "published": bool}.

    - параметры — DEFAULT_CONFIG, поверх которого применяются ключи из config;
    - каталоги: workdir/data, workdir/runs/YYYY-MM-DD, workdir/registry;
    - для каждого дня обучающего и проверочного окон: extract, затем validate. Ошибка проверки
      (DataValidationError) прерывает запуск: модель не обучается и не публикуется;
    - затем train, evaluate, publish с порогом config["max_mae_ratio"];
    - каталог запуска после успешного выполнения содержит ровно три файла: model.joblib, metrics.json,
      metadata.json; файлы прошлой попытки за ту же дату заменяются;
    - metadata.json (json.dumps(..., indent=2, sort_keys=True, ensure_ascii=False)) содержит:
        "run_date" — "YYYY-MM-DD";
        "config" — использованные параметры;
        "train_period", "valid_period" — пары ["YYYY-MM-DD", "YYYY-MM-DD"]: первый и последний день окна;
        "data_hash" — sha256 (hex) от содержимого всех партиций обоих окон, прочитанных как байты
                      и сцепленных в порядке возрастания даты;
        "metrics" — результат evaluate; "published" — результат publish;
        "versions" — версии "python", "numpy", "pandas", "scikit-learn", "joblib";
    - идемпотентность: повторный запуск за ту же дату с тем же источником оставляет в workdir тот же набор
      файлов, а metrics.json и metadata.json совпадают байт в байт. Поэтому в метаданных нет времени
      запуска: оно принадлежит логам, а не артефактам.
    """
    raise NotImplementedError


def sales_source(day: date) -> pd.DataFrame:
    """Источник для настоящего запуска: генератор темы 2.1 (файл sales_data.py должен быть рядом)."""
    sales = _all_sales()
    return sales[sales["date"] == pd.Timestamp(day)]


@cache
def _all_sales() -> pd.DataFrame:
    from sales_data import make_sales

    return make_sales()


def main() -> None:
    parser = argparse.ArgumentParser(description="Пайплайн прогноза продаж за одну дату")
    parser.add_argument("--date", type=date.fromisoformat, required=True, help="дата запуска, YYYY-MM-DD")
    parser.add_argument("--workdir", type=Path, default=Path("work"))
    args = parser.parse_args()
    result = run(args.date, args.workdir, sales_source)
    print(f"{args.date}: published={result['published']} metrics={result['metrics']}")


if __name__ == "__main__":
    main()
