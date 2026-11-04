import tomllib
from pathlib import Path

import kiyal

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


def test_installed_version_matches_pyproject() -> None:
    # A mismatch means the environment holds a stale install of the package.
    with PYPROJECT.open("rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    assert kiyal.__version__ == expected
