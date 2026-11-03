"""Space-Track client: credentials hygiene, policy ledger, cache, conversion. No network."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from leo_decay import spacetrack as stmod
from leo_decay.spacetrack import (
    PolicyViolation, SpaceTrack, SpaceTrackError, get_credentials, load_tle_toml,
    to_dataframe, write_tle_toml,
)

FAKE_USER, FAKE_PASS = "user@example.invalid", "s3cret-PASSWORD-xyz"


def _rec(norad, epoch, mm, created="2026-10-20T00:00:00", name="OBJ"):
    return {"NORAD_CAT_ID": str(norad), "OBJECT_NAME": name, "EPOCH": epoch,
            "MEAN_MOTION": str(mm), "ECCENTRICITY": "0.0002", "INCLINATION": "28.47",
            "RA_OF_ASC_NODE": "100.0", "ARG_OF_PERICENTER": "50.0", "MEAN_ANOMALY": "10.0",
            "BSTAR": "0.0001", "CREATION_DATE": created,
            "TLE_LINE1": f"1 {norad}U line1", "TLE_LINE2": f"2 {norad} line2"}


class FakeClient:
    """Stands in for spacetrack.SpaceTrackClient."""
    calls = []

    def __init__(self, response=None, error=None):
        self.response, self.error = response, error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _call(self, cls, **query):
        FakeClient.calls.append((cls, query))
        if self.error:
            raise self.error
        return self.response

    def gp(self, **q):
        return self._call("gp", **q)

    def gp_history(self, **q):
        return self._call("gp_history", **q)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACETRACK_USER", FAKE_USER)
    monkeypatch.setenv("SPACETRACK_PASSWORD", FAKE_PASS)
    monkeypatch.setattr(stmod, "DEFAULT_ENV_FILE", tmp_path / "no.env")
    FakeClient.calls = []


class Clock:
    def __init__(self):
        self.t = datetime(2026, 11, 3, 12, 17, tzinfo=timezone.utc)

    def __call__(self):
        return self.t


def _client(tmp_path, response=None, error=None, clock=None):
    return SpaceTrack(cache_dir=tmp_path / "cache",
                      client_factory=lambda: FakeClient(response, error),
                      now=clock or Clock())


# ------------------------------------------------------------------ credentials
def test_missing_credentials_message_has_no_values(monkeypatch, tmp_path):
    monkeypatch.delenv("SPACETRACK_PASSWORD")
    with pytest.raises(SpaceTrackError) as exc:
        get_credentials(tmp_path / "no.env")
    assert "SPACETRACK_PASSWORD" in str(exc.value)
    assert FAKE_USER not in str(exc.value)


def test_credentials_available_returns_only_booleans(tmp_path):
    status = stmod.credentials_available(tmp_path / "no.env")
    assert status == {"SPACETRACK_USER": True, "SPACETRACK_PASSWORD": True}


def test_errors_and_repr_never_contain_credentials(tmp_path):
    err = RuntimeError(f"login failed for {FAKE_USER} with {FAKE_PASS}")
    st = _client(tmp_path, error=err)
    with pytest.raises(SpaceTrackError) as exc:
        st.fetch_latest_gp([20580])
    msg = str(exc.value)
    assert FAKE_USER not in msg and FAKE_PASS not in msg and "***" in msg
    assert exc.value.__cause__ is None and exc.value.__suppress_context__
    assert FAKE_PASS not in repr(st) and FAKE_USER not in repr(st)


def test_cache_files_contain_no_credentials(tmp_path):
    st = _client(tmp_path, response=[_rec(20580, "2026-10-20T00:00:00", 15.2)])
    st.fetch_latest_gp([20580])
    st.fetch_gp_history(20580, date(2026, 1, 1), date(2026, 11, 3))
    for f in (tmp_path / "cache").iterdir():
        text = f.read_text(encoding="utf-8")
        assert FAKE_USER not in text and FAKE_PASS not in text


# ------------------------------------------------------------------ GP policy
def test_gp_batched_in_one_request(tmp_path):
    st = _client(tmp_path, response=[_rec(i, "2026-11-03T00:00:00", 15.0) for i in (20580, 25338)])
    entry = st.fetch_latest_gp([25544, 20580, 25338, 20580])
    assert len(FakeClient.calls) == 1
    cls, q = FakeClient.calls[0]
    assert cls == "gp" and q["norad_cat_id"] == [20580, 25338, 25544]
    assert entry.path.exists() and entry.meta["records"] == 2


def test_gp_at_most_once_per_hour(tmp_path):
    clock = Clock()
    st = _client(tmp_path, response=[], clock=clock)
    st.fetch_latest_gp([20580])
    clock.t += timedelta(minutes=59)
    with pytest.raises(PolicyViolation, match="one GP query per hour"):
        st.fetch_latest_gp([20580])
    clock.t += timedelta(minutes=2)
    st.fetch_latest_gp([20580])
    assert len(FakeClient.calls) == 2


def test_gp_limit_persists_across_instances(tmp_path):
    clock = Clock()
    _client(tmp_path, response=[], clock=clock).fetch_latest_gp([20580])
    with pytest.raises(PolicyViolation):
        _client(tmp_path, response=[], clock=clock).fetch_latest_gp([20580])


def test_failed_gp_attempt_still_counts(tmp_path):
    clock = Clock()
    with pytest.raises(SpaceTrackError):
        _client(tmp_path, error=RuntimeError("boom"), clock=clock).fetch_latest_gp([20580])
    with pytest.raises(PolicyViolation):
        _client(tmp_path, response=[], clock=clock).fetch_latest_gp([20580])


# ------------------------------------------------------------------ GP_HISTORY policy
def test_history_downloaded_once_then_cached(tmp_path):
    recs = [_rec(20580, "2026-10-15T01:00:00", 15.20), _rec(20580, "2026-10-16T01:00:00", 15.21)]
    st = _client(tmp_path, response=recs)
    first = st.fetch_gp_history(20580, date(2026, 1, 1), date(2026, 11, 3))
    again = st.fetch_gp_history(20580, date(2019, 1, 1), date(2026, 11, 3))
    assert len(FakeClient.calls) == 1
    cls, q = FakeClient.calls[0]
    assert cls == "gp_history" and q["norad_cat_id"] == 20580 and q["orderby"] == "epoch asc"
    assert again.records == first.records == recs
    assert again.meta["epoch_start"] == "2026-01-01"


def test_history_refused_if_cache_lost(tmp_path):
    st = _client(tmp_path, response=[])
    entry = st.fetch_gp_history(20580, date(2026, 1, 1), date(2026, 11, 3))
    entry.path.unlink()
    with pytest.raises(PolicyViolation, match="do not download it again"):
        st.fetch_gp_history(20580, date(2026, 1, 1), date(2026, 11, 3))
    assert len(FakeClient.calls) == 1


def test_history_other_object_allowed(tmp_path):
    st = _client(tmp_path, response=[])
    st.fetch_gp_history(20580, date(2026, 1, 1), date(2026, 11, 3))
    st.fetch_gp_history(25338, date(2026, 1, 1), date(2026, 11, 3))
    assert len(FakeClient.calls) == 2


def test_load_cached_history_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        _client(tmp_path).load_cached_history(20580)


# ------------------------------------------------------------------ conversion & toml
def test_to_dataframe_types_and_dedupe():
    recs = [_rec(20580, "2026-10-16T00:00:00", 15.3),
            _rec(20580, "2026-10-15T00:00:00", 15.20, created="2026-10-15T05:00:00"),
            _rec(20580, "2026-10-15T00:00:00", 15.25, created="2026-10-15T09:00:00")]
    df = to_dataframe(recs)
    assert len(df) == 2
    assert df["EPOCH"].dt.tz is not None and df["EPOCH"].is_monotonic_increasing
    assert df.loc[0, "MEAN_MOTION"] == 15.25                   # newest re-publication kept
    assert df["INCLINATION"].dtype.kind == "f"


def test_tle_toml_round_trip(tmp_path):
    df = to_dataframe([_rec(20580, "2026-11-01T00:00:00", 15.3, name="HST"),
                       _rec(20580, "2026-11-02T00:00:00", 15.31, name="HST"),
                       _rec(25544, "2026-11-02T06:00:00", 15.5, name="ISS")])
    path = write_tle_toml(df, {20580: "hubble", 25544: "iss"}, tmp_path / "tle.toml",
                          "2026-11-03T12:17:00+00:00")
    data = load_tle_toml(path)
    assert set(data) == {"hubble", "iss"}
    h = data["hubble"]
    assert h["norad_id"] == 20580 and h["mean_motion_rev_day"] == 15.31   # latest epoch
    assert h["epoch_utc"].startswith("2026-11-02T00:00:00")
    assert h["inclination_deg"] == 28.47 and h["line1"].startswith("1 20580")


def test_example_toml_matches_generated_format(tmp_path):
    from pathlib import Path
    example = load_tle_toml(Path(stmod.PROJECT_ROOT) / "config" / "tle.example.toml")
    df = to_dataframe([_rec(20580, "2026-11-01T00:00:00", 15.3)])
    generated = load_tle_toml(write_tle_toml(df, {20580: "hubble"}, tmp_path / "t.toml", "x"))
    assert set(example["hubble"]) == set(generated["hubble"])
    assert set(example) == {"hubble", "iss", "noaa15"}
