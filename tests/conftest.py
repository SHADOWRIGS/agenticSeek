import os
import pytest


@pytest.fixture(autouse=True)
def set_work_dir(tmp_path):
    old = os.environ.get("WORK_DIR")
    os.environ["WORK_DIR"] = str(tmp_path)
    yield
    if old is None:
        os.environ.pop("WORK_DIR", None)
    else:
        os.environ["WORK_DIR"] = old
