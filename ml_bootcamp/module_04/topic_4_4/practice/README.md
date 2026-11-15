# Практика 4.4.4: материалы

Постановка задачи: [lesson_4_4_4.md](../lesson_4_4_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `ensemble_lab.py` | заготовка стенда: `out_of_fold_predictions`, `error_correlation`, `average`, `rank_average`, `fit_blend_weights`, `fit_stacker`, `predict_stacker`, `evaluate_ensembles` |
| `tests/test_ensemble_lab.py` | 70 тестов: примеры с известным ответом, отсутствие утечки целевой переменной, свойства весов, честность оценки ансамблей, воспроизводимость |

Стенд не зависит от данных и от библиотек бустинга: функции принимают модели scikit-learn и таблицы pandas.
Тесты работают на маленьких синтетических данных (300 объектов, логистическая регрессия, дерево, kNN)
и не обращаются к сети.

## Запуск

Версии, на которых проверены заготовка и тесты: Python 3.12, `scikit-learn 1.9.1`, `pandas 3.1.1`,
`numpy 2.5.4`, `pytest 9.2.0`.

```bash
uv add numpy pandas scikit-learn
uv add --dev pytest ruff
uv run pytest -q          # на заготовке не проходит ни один тест: 33 failed, 37 errors
```

По шагам:

```bash
uv run pytest -q -k TestOutOfFoldPredictions
uv run pytest -q -k TestErrorCorrelation
uv run pytest -q -k "TestAverage or TestRankAverage"
uv run pytest -q -k TestFitBlendWeights
uv run pytest -q -k TestStacker
uv run pytest -q -k TestEvaluateEnsembles
```

Общие фикстуры объявлены на уровне модуля: `data` (задача классификации), `folds` (разбиение),
`predictions` (результат вашей `out_of_fold_predictions`), `calibrated` и `noise_models` (данные для
проверки метамодели и честности оценки). Фикстура `predictions` вызывает вашу функцию, поэтому до её
реализации тесты, которые от неё зависят, завершаются ошибкой на этапе подготовки (в отчёте pytest —
`ERROR`, а не `FAILED`).

Эталонное решение проходит все тесты за 22 секунды. Самые долгие — два теста честности оценки на
15 моделях-шумах (по 5 секунд): жадный подбор весов вычисляет метрику 50 × 15 раз на каждом из пяти
разбиений. Полный прогон дольше минуты означает, что подбор весов пересчитывает среднее по набору заново
на каждом шаге вместо того, чтобы хранить накопленную сумму.

Строгая проверка — без предупреждений библиотек:

```bash
uv run python -W error -m pytest -q
```

Проверка стиля:

```bash
uv run ruff check .
uv run ruff format --check ensemble_lab.py tests/test_ensemble_lab.py
```

## Данные для второй части

Демо-данные — те же, что в практике 2.3.4: генератор `make_demo_data.py` лежит в
[`module_02/topic_2_3/practice/`](../../../module_02/topic_2_3/practice/). Если практика 2.3.4 у вас
выполнена, файл уже есть в `projects/p_2_3_kaggle/practice/`. Скопируйте его в папку проекта без
изменений или добавьте путь к нему в `sys.path`:

```python
import sys

sys.path.insert(0, "../p_2_3_kaggle/practice")
from make_demo_data import TARGET, make_demo, score_submission

train, test, solution = make_demo()      # 8 000 и 4 000 строк; solution — «скрытые» ответы теста
```

`score_submission(submission, solution)` возвращает ROC-AUC на публичной и приватной частях теста —
локальная замена лидерборда.

Для второй части понадобятся библиотеки, которых нет в зависимостях стенда:

```bash
uv add lightgbm catboost optuna
```

## Замечания

- Тест `test_own_target_does_not_leak` использует `KFold`, а не `StratifiedKFold`: у стратифицированного
  разбиения состав фолдов зависит от ответов, и замена одного ответа изменила бы сами фолды.
- Тест `test_same_folds_for_all_models` передаёт разбиение с перемешиванием без `random_state`. Если
  вызывать `cv.split` отдельно для каждой модели, столбцы таблицы окажутся посчитанными на разных фолдах.
- CatBoost по умолчанию создаёт каталог `catboost_info`, а Optuna с `storage="sqlite:///..."` — файл базы.
  Ни то, ни другое, ни файлы данных в репозиторий не добавляйте.
