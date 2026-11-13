# Практика 3.3.4: материалы

Постановка задачи: [lesson_3_3_4.md](../lesson_3_3_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `segmentation.py` | заготовка: `build_rfm_features`, `smooth_shares`, `preprocess`, `evaluate_k`, `stability_score`, `cluster_profiles`, `assign_segment` |
| `tests/test_segmentation.py` | тесты, описывающие ожидаемое поведение функций |
| `tests/conftest.py` | ограничивает вычисления одним потоком на время тестов |

Файл `customers_data.py` должен лежать на уровень выше папки `practice` — так же, как в материалах курса.

## Запуск

```bash
uv add pandas numpy scikit-learn scipy
uv add --dev pytest
cd practice
uv run pytest -q          # на заготовке все тесты падают
```

Удобно работать по одной функции:

```bash
uv run pytest -q -k TestBuildRfmFeatures
uv run pytest -q -k TestSmoothShares
uv run pytest -q -k TestPreprocess
uv run pytest -q -k TestEvaluateK
uv run pytest -q -k TestStabilityScore
uv run pytest -q -k TestClusterProfiles
uv run pytest -q -k TestAssignSegment
uv run pytest -q -k TestPipelineOnCourseData   # вся цепочка на данных курса
```

Полный прогон на правильном решении занимает несколько секунд. Тесты класса `TestPipelineOnCourseData`
используют истинные сегменты генератора — в своём анализе обращайтесь к ним только в последнем пункте
плана.
