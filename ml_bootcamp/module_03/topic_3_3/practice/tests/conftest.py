import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(scope="session", autouse=True)
def single_thread():
    # Тесты сотни раз обучают k-means на маленьких выборках. В несколько потоков это не быстрее,
    # а на загруженной машине — в десятки раз медленнее: потоки ждут друг друга.
    with threadpool_limits(limits=1):
        yield
