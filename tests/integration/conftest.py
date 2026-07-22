from pathlib import Path

import pytest

_INTEGRATION_ROOT = Path(__file__).resolve().parent


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    _ = config
    for item in items:
        item_path = Path(str(item.path)).resolve()
        if item_path.is_relative_to(_INTEGRATION_ROOT):
            item.add_marker(pytest.mark.integration)
