from datetime import date

import pytest

from leo_decay import spacetrack


def test_fetch_is_stubbed():
    with pytest.raises(NotImplementedError):
        spacetrack.fetch_tles(20580, date(2026, 10, 15), date(2026, 11, 19))


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASS", raising=False)
    with pytest.raises(RuntimeError):
        spacetrack.get_credentials()
