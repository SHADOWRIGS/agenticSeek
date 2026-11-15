# Практика 3.4.4: материалы

Постановка задачи: [lesson_3_4_4.md](../lesson_3_4_4.md).

## Состав

| Файл | Назначение |
|---|---|
| `recsys.py` | заготовка: `time_split`, `build_interaction_matrix`, `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `catalog_coverage`, `PopularRecommender`, `ItemKNNRecommender`, `ALSRecommender`, `recommend_ids` |
| `tests/test_recsys.py` | тесты, описывающие ожидаемое поведение |

Файл `shop_interactions.py` должен лежать на уровень выше папки `practice` — так же, как в материалах курса.

## Запуск

```bash
uv add pandas numpy scipy
uv add --dev pytest
cd practice
uv run pytest -q          # на заготовке все тесты падают с NotImplementedError
```

Удобно работать по одной части:

```bash
uv run pytest -q -k TestTimeSplit
uv run pytest -q -k TestBuildInteractionMatrix
uv run pytest -q -k TestMetrics
uv run pytest -q -k TestPopularRecommender
uv run pytest -q -k "TestPersonalModelOnBlocks or TestItemKNNDetails or TestALSDetails"
uv run pytest -q -k TestOnShopData
```

## Одна персональная модель из двух

Обязательна одна из моделей — `ItemKNNRecommender` или `ALSRecommender`. Пока не реализована ни одна,
тесты персональных моделей падают. Как только заработает `fit` одной из них, тесты второй помечаются как
пропущенные (`skipped`): итог «45 passed, 18 skipped» означает, что работа принята с одной моделью,
«63 passed» — с обеими.

Полный прогон на правильном решении занимает несколько секунд. Если тесты `TestOnShopData` идут
минуты или падает `test_speed` — в обучении или в рекомендациях остались циклы Python по парам
«пользователь–товар» или «товар–товар».

Проверка стиля запускается из этой же папки:

```bash
uv run ruff check .
uv run ruff format --check recsys.py tests/test_recsys.py
```
