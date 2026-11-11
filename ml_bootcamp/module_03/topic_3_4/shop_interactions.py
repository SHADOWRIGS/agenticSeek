"""Синтетический интернет-магазин для уроков темы 3.4: пользователи, товары и журнал событий.

Использование:  from shop_interactions import make_shop
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

START = pd.Timestamp("2025-01-01")
N_DAYS = 180  # журнал покрывает период с 2025-01-01 по 2025-06-29
N_FACTORS = 8
NOVELTY_DAYS = 14  # первые две недели после выхода товар показывают чаще
NOVELTY_BOOST = 3.0

# категория: (типы товаров, типичная цена в рублях, доля каталога)
CATALOG = {
    "Электроника": (["наушники", "смартфон", "планшет", "колонка", "часы"], 15000, 0.14),
    "Бытовая техника": (["чайник", "пылесос", "блендер", "утюг", "кофеварка"], 6000, 0.08),
    "Книги": (["роман", "детектив", "учебник", "комикс", "фантастика"], 600, 0.16),
    "Одежда": (["футболка", "куртка", "джинсы", "платье", "свитер"], 3000, 0.15),
    "Обувь": (["кроссовки", "ботинки", "сандалии", "туфли", "сапоги"], 4500, 0.07),
    "Спорт": (["гантели", "коврик", "велосипед", "палатка", "рюкзак"], 5000, 0.08),
    "Дом": (["плед", "лампа", "посуда", "полка", "шторы"], 2000, 0.10),
    "Красота": (["крем", "шампунь", "парфюм", "помада", "маска"], 1200, 0.09),
    "Детям": (["конструктор", "кукла", "пазл", "самокат", "коляска"], 2500, 0.08),
    "Продукты": (["кофе", "чай", "шоколад", "орехи", "специи"], 500, 0.05),
}
BRANDS = (
    "Altera Bravo Corvus Delta Evora Fjord Gamma Helix Indigo Juno Kappa Lumen Mistral Nord Orion Polar "
    "Quanta Rondo Sigma Terra Ultra Vega Wave Xenon Yukon Zenit Aster Brio Cedar Dune Ember Flux Grove Halo "
    "Iris Jade Kite Lotus Mango Nova"
).split()
# слово попадает в описание, если соответствующий скрытый фактор товара выражен сильно
STYLE_WORDS = ["премиум", "эко", "компактный", "яркий", "классика", "походный", "подарочный", "профи"]
FILLER_WORDS = ["новинка", "хит", "скидка", "качество", "доставка", "оригинал", "гарантия", "выбор"]


@dataclass(frozen=True)
class Shop:
    """Таблицы магазина. hidden — то, чего в реальных данных не бывает: истинные скрытые факторы."""

    users: pd.DataFrame  # user_id, registered_at
    items: pd.DataFrame  # item_id, category, brand, price, description, released_at
    events: pd.DataFrame  # user_id, item_id, event ("view" | "purchase"), timestamp, rating
    hidden: dict[str, np.ndarray]  # user_factors, item_factors — строки в порядке таблиц users и items


def _make_items(rng: np.random.Generator, n_items: int, centers: np.ndarray):
    categories = list(CATALOG)
    shares = np.array([CATALOG[c][2] for c in categories])
    cat = rng.choice(len(categories), size=n_items, p=shares / shares.sum())
    kind = rng.integers(0, 5, n_items)
    brand = cat * 4 + rng.integers(0, 4, n_items)  # у каждой категории четыре своих бренда

    kind_shift = rng.normal(0, 0.5, (len(categories), 5, N_FACTORS))
    brand_shift = rng.normal(0, 0.4, (len(BRANDS), N_FACTORS))
    factors = centers[cat] + kind_shift[cat, kind] + brand_shift[brand] + rng.normal(0, 0.5, (n_items, N_FACTORS))

    # цена связана с первым скрытым фактором («премиальность»)
    base_price = np.array([CATALOG[c][1] for c in categories])[cat]
    price = base_price * np.exp(0.45 * factors[:, 0] + rng.normal(0, 0.25, n_items))
    price = np.maximum(np.round(price, -1), 50)

    descriptions = []
    for i in range(n_items):
        words = [CATALOG[categories[cat[i]]][0][kind[i]], BRANDS[brand[i]]]
        words += [word for word, value in zip(STYLE_WORDS, factors[i], strict=True) if value > 0.9]
        words += list(rng.choice(FILLER_WORDS, size=2, replace=False))
        descriptions.append(" ".join(words))

    # 65 % каталога существует к началу журнала, остальные товары появляются в течение периода
    release_day = np.where(rng.random(n_items) < 0.65, rng.uniform(-365, 0, n_items), rng.uniform(0, N_DAYS, n_items))
    items = pd.DataFrame(
        {
            "item_id": 5000 + rng.permutation(n_items),
            "category": np.array(categories)[cat],
            "brand": np.array(BRANDS)[brand],
            "price": price,
            "description": descriptions,
            "released_at": (START + pd.to_timedelta(np.floor(release_day), unit="D")),
        }
    )
    return items, factors, release_day


def _make_users(rng: np.random.Generator, n_users: int, centers: np.ndarray):
    # у пользователя одна-три любимые категории; его вектор — смесь их центров плюс личные вкусы
    n_categories = len(centers)
    weights = np.zeros((n_users, n_categories))
    for u in range(n_users):
        liked = rng.choice(n_categories, size=rng.integers(1, 4), replace=False)
        weights[u, liked] = rng.dirichlet(np.ones(len(liked)))
    factors = weights @ centers + rng.normal(0, 0.45, (n_users, N_FACTORS))

    register_day = np.where(rng.random(n_users) < 0.55, rng.uniform(-365, 0, n_users), rng.uniform(0, N_DAYS, n_users))
    users = pd.DataFrame(
        {
            "user_id": 100000 + rng.permutation(n_users),
            "registered_at": START + pd.to_timedelta(np.floor(register_day), unit="D"),
        }
    )
    return users, factors, register_day


def make_shop(n_users: int = 3000, n_items: int = 2000, seed: int = 42) -> Shop:
    """Генерирует магазин: по умолчанию 3 000 пользователей, 2 000 товаров, около 86 тысяч событий.

    Как устроены данные:

    - у каждого пользователя и товара есть вектор из 8 скрытых факторов; «близость» пары — их скалярное
      произведение. Вектор товара складывается из центра категории, сдвигов типа товара и бренда
      и индивидуальной части, которой нет ни в одном признаке товара;
    - пользователь просматривает товары тем чаще, чем выше близость и чем популярнее товар (популярность
      распределена с длинным хвостом). Каждый пятый просмотр случаен: зависит только от популярности;
    - вероятность покупки после просмотра растёт с близостью; покупка получает оценку от 1 до 5,
      зависящую от близости, строгости пользователя и качества товара;
    - пара (пользователь, товар) встречается в журнале не более одного раза: просмотр и, возможно,
      покупка через несколько минут после него;
    - активность пользователей различается в десятки раз; 45 % пользователей регистрируются и 35 % товаров
      выходят уже в течение периода; новые товары первые две недели показываются втрое чаще.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, (len(CATALOG), N_FACTORS))
    items, item_factors, release_day = _make_items(rng, n_items, centers)
    users, user_factors, register_day = _make_users(rng, n_users, centers)

    popularity = rng.normal(0, 1.1, n_items)
    user_strictness = rng.normal(0, 0.4, n_users)
    item_quality = rng.normal(0, 0.4, n_items)
    affinity = user_factors @ item_factors.T
    affinity = (affinity - affinity.mean()) / affinity.std()

    # «экспозиция» пары: сколько дней товар доступен пользователю, дни новизны считаются с весом NOVELTY_BOOST
    start = np.maximum(np.maximum(register_day, 0)[:, None], release_day[None, :])
    boost_end = np.minimum(np.maximum(release_day + NOVELTY_DAYS, 0), N_DAYS)[None, :]
    boosted = np.clip(boost_end - start, 0, None)
    regular = N_DAYS - np.maximum(start, boost_end)
    exposure = NOVELTY_BOOST * boosted + regular
    window = (N_DAYS - np.maximum(register_day, 0))[:, None]

    activity = rng.lognormal(np.log(22), 0.9, n_users) * window[:, 0] / N_DAYS
    n_views = np.clip(np.round(activity), 1, n_items // 10).astype(int)
    n_random = rng.binomial(n_views, 0.2)

    with np.errstate(divide="ignore"):
        log_exposure = np.log(exposure / window)
    # выбор без возвращения с вероятностями softmax: прибавляем шум Гумбеля и берём наибольшие
    taste_score = 1.6 * affinity + popularity + log_exposure + rng.gumbel(size=affinity.shape)
    random_score = popularity + log_exposure + rng.gumbel(size=affinity.shape)
    rank = np.arange(n_items)[None, :]
    taste_order = np.argsort(-taste_score, axis=1)
    taste_mask = rank < (n_views - n_random)[:, None]
    row_taste, col_taste = np.nonzero(taste_mask)
    col_taste = taste_order[row_taste, col_taste]
    random_score[row_taste, col_taste] = -np.inf
    random_order = np.argsort(-random_score, axis=1)
    row_random, col_random = np.nonzero(rank < n_random[:, None])
    col_random = random_order[row_random, col_random]
    u = np.concatenate([row_taste, row_random])
    i = np.concatenate([col_taste, col_random])
    keep = exposure[u, i] > 0
    u, i = u[keep], i[keep]

    # время просмотра: в «дни новизны» с вероятностью, пропорциональной их вкладу в экспозицию
    in_boost = rng.random(len(u)) < NOVELTY_BOOST * boosted[u, i] / exposure[u, i]
    low = np.where(in_boost, start[u, i], np.maximum(start[u, i], boost_end[0, i]))
    high = np.where(in_boost, np.maximum(start[u, i], boost_end[0, i]), N_DAYS)
    view_day = rng.uniform(low, high)

    pair_affinity = affinity[u, i]
    bought = rng.random(len(u)) < 1 / (1 + np.exp(-(-2.3 + 1.2 * pair_affinity)))
    rating = 2.5 + 0.7 * pair_affinity + user_strictness[u] + item_quality[i] + rng.normal(0, 0.7, len(u))
    rating = np.clip(np.rint(rating), 1, 5)
    buy_day = np.minimum(view_day + rng.exponential(20 / 1440, len(u)), N_DAYS - 1e-4)

    def to_time(day: np.ndarray) -> pd.Series:
        return pd.Series(START + pd.to_timedelta(np.floor(day * 86400), unit="s"))

    views = pd.DataFrame(
        {
            "user_id": users["user_id"].to_numpy()[u],
            "item_id": items["item_id"].to_numpy()[i],
            "event": "view",
            "timestamp": to_time(view_day),
            "rating": np.nan,
        }
    )
    purchases = views[bought].assign(event="purchase", timestamp=to_time(buy_day)[bought], rating=rating[bought])
    events = pd.concat([views, purchases]).sort_values(["timestamp", "event"], kind="stable", ignore_index=True)
    hidden = {"user_factors": user_factors, "item_factors": item_factors}
    return Shop(users=users, items=items, events=events, hidden=hidden)
