"""Space-Track.org client with policy enforcement and an on-disk cache.

Credentials come from the environment (``SPACETRACK_USER``,
``SPACETRACK_PASSWORD``), loaded from ``.env`` with python-dotenv. They are
never printed, logged, stored on objects, or included in error messages.

Space-Track API Use Guidelines (checked 2026-11-03):
  * < 30 requests/minute and < 300 requests/hour overall
    -> enforced per process by the ``spacetrack`` package itself;
  * GP class: at most once per hour, batch NORAD IDs into one request
    -> enforced across runs by the request ledger below;
  * GP_HISTORY: "1 / lifetime" per object -- "Once you download an object's
    history, you need to store it on your own servers; do not download it
    again." -> enforced by the ledger; responses are cached permanently.

All raw responses are cached unmodified under ``data/raw/spacetrack/``
(gitignored), with a metadata sidecar per file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "spacetrack"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
USER_VAR, PASSWORD_VAR = "SPACETRACK_USER", "SPACETRACK_PASSWORD"

GP_MIN_INTERVAL = timedelta(hours=1)

NUMERIC_COLUMNS = ["MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
                   "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR", "MEAN_MOTION_DOT",
                   "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS", "PERIAPSIS"]


class SpaceTrackError(RuntimeError):
    """Client errors. Messages never contain credentials."""


class PolicyViolation(SpaceTrackError):
    """A request that Space-Track's usage guidelines say not to make."""


# --------------------------------------------------------------------- credentials
def credentials_available(env_file: str | Path = DEFAULT_ENV_FILE) -> dict[str, bool]:
    """Load ``.env`` (if present) and report which variables are set. Never returns values."""
    _load_env(env_file)
    return {USER_VAR: bool(os.environ.get(USER_VAR)),
            PASSWORD_VAR: bool(os.environ.get(PASSWORD_VAR))}


def get_credentials(env_file: str | Path = DEFAULT_ENV_FILE) -> tuple[str, str]:
    """Return (user, password). Callers must not print or store them."""
    _load_env(env_file)
    user, password = os.environ.get(USER_VAR), os.environ.get(PASSWORD_VAR)
    missing = [name for name, v in ((USER_VAR, user), (PASSWORD_VAR, password)) if not v]
    if missing:
        raise SpaceTrackError(f"Missing environment variable(s): {', '.join(missing)} "
                              f"(set them in .env).")
    return user, password


def _load_env(env_file) -> None:
    from dotenv import load_dotenv
    load_dotenv(env_file, override=False)


def _scrub(text: str) -> str:
    """Remove any credential value that might appear in a message."""
    for var in (USER_VAR, PASSWORD_VAR):
        value = os.environ.get(var)
        if value:
            text = text.replace(value, "***")
    return text


# --------------------------------------------------------------------- ledger
class RequestLedger:
    """Persistent record of requests, so policy holds across separate runs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: list[dict] = (json.loads(self.path.read_text(encoding="utf-8"))
                                    if self.path.exists() else [])

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")

    def record(self, request_class: str, norad_ids: Iterable[int], utc: datetime, **detail) -> None:
        self.entries.append({"class": request_class, "norad_ids": sorted(int(i) for i in norad_ids),
                             "utc": utc.isoformat(timespec="seconds"), **detail})
        self._save()

    def last(self, request_class: str) -> datetime | None:
        times = [datetime.fromisoformat(e["utc"]) for e in self.entries
                 if e["class"] == request_class]
        return max(times) if times else None

    def history_downloaded(self, norad_id: int) -> bool:
        return any(e["class"] == "gp_history" and int(norad_id) in e["norad_ids"]
                   for e in self.entries)


# --------------------------------------------------------------------- client
@dataclass(frozen=True)
class CacheEntry:
    path: Path
    records: list[dict]
    meta: dict


def _default_client_factory():
    from spacetrack import SpaceTrackClient
    user, password = get_credentials()
    return SpaceTrackClient(identity=user, password=password)


class SpaceTrack:
    """Policy-aware wrapper. ``client_factory`` is injectable for tests."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR,
                 client_factory: Callable = _default_client_factory,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.cache_dir = Path(cache_dir)
        self.ledger = RequestLedger(self.cache_dir / "ledger.json")
        self._client_factory = client_factory
        self._now = now

    def __repr__(self) -> str:                      # no credentials, ever
        return f"SpaceTrack(cache_dir={str(self.cache_dir)!r})"

    # -- internals
    def _request(self, request_class: str, **query) -> list[dict]:
        try:
            with self._client_factory() as st:
                data = getattr(st, request_class)(**query)
        except Exception as exc:                    # re-raise without credentials or chaining
            raise SpaceTrackError(_scrub(f"{request_class} request failed: "
                                         f"{type(exc).__name__}: {exc}")) from None
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, list):
            raise SpaceTrackError(f"Unexpected {request_class} response type: {type(data).__name__}")
        return data

    def _store(self, name: str, records: list[dict], meta: dict) -> CacheEntry:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{name}.json"
        path.write_text(json.dumps(records, indent=1), encoding="utf-8")
        meta = {**meta, "file": path.name, "records": len(records),
                "retrieved_utc": self._now().isoformat(timespec="seconds"),
                "source": "Space-Track.org"}
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return CacheEntry(path, records, meta)

    @staticmethod
    def _read(path: Path) -> CacheEntry:
        return CacheEntry(path, json.loads(path.read_text(encoding="utf-8")),
                          json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8")))

    # -- GP (latest elements)
    def fetch_latest_gp(self, norad_ids: Iterable[int]) -> CacheEntry:
        """One batched GP request (policy: at most once per hour)."""
        ids = sorted({int(i) for i in norad_ids})
        last = self.ledger.last("gp")
        if last is not None and self._now() - last < GP_MIN_INTERVAL:
            wait = GP_MIN_INTERVAL - (self._now() - last)
            raise PolicyViolation(f"GP was queried at {last:%H:%M} UTC; Space-Track allows one GP "
                                  f"query per hour. Use the cache or retry in "
                                  f"{int(wait.total_seconds() // 60) + 1} min.")
        self.ledger.record("gp", ids, self._now())  # attempts count against the limit too
        records = self._request("gp", norad_cat_id=ids, orderby="norad_cat_id")
        stamp = self._now().strftime("%Y%m%dT%H%M%SZ")
        return self._store(f"gp_latest_{stamp}", records, {"class": "gp", "norad_ids": ids})

    def latest_cached_gp(self) -> CacheEntry | None:
        files = sorted(p for p in self.cache_dir.glob("gp_latest_*.json")
                       if not p.name.endswith(".meta.json"))
        return self._read(files[-1]) if files else None

    # -- GP_HISTORY
    def history_path(self, norad_id: int) -> Path:
        return self.cache_dir / f"gp_history_{int(norad_id)}.json"

    def fetch_gp_history(self, norad_id: int, start: date, end: date,
                         allow_repeat: bool = False) -> CacheEntry:
        """Download an object's history once; afterwards always served from cache.

        Policy: GP_HISTORY is "1 / lifetime" per object. A second download is
        refused unless ``allow_repeat=True`` is passed deliberately.
        """
        norad_id = int(norad_id)
        if self.history_path(norad_id).exists() and not allow_repeat:
            return self._read(self.history_path(norad_id))
        if self.ledger.history_downloaded(norad_id) and not allow_repeat:
            raise PolicyViolation(
                f"GP_HISTORY for {norad_id} was already downloaded (ledger) but the cache file is "
                f"missing. Space-Track: 'do not download it again'. Restore the cache file, or "
                f"pass allow_repeat=True only if you have a good reason.")
        from spacetrack import operators as op
        self.ledger.record("gp_history", [norad_id], self._now(), start=str(start), end=str(end))
        records = self._request("gp_history", norad_cat_id=norad_id,
                                epoch=op.inclusive_range(start, end), orderby="epoch asc")
        return self._store(self.history_path(norad_id).stem, records,
                           {"class": "gp_history", "norad_ids": [norad_id],
                            "epoch_start": str(start), "epoch_end": str(end)})

    def load_cached_history(self, norad_id: int) -> CacheEntry:
        path = self.history_path(norad_id)
        if not path.exists():
            raise FileNotFoundError(f"No cached GP_HISTORY for {norad_id} at {path}")
        return self._read(path)


# --------------------------------------------------------------------- conversion
def to_dataframe(records: list[dict]) -> pd.DataFrame:
    """GP/GP_HISTORY JSON records -> DataFrame with UTC ``EPOCH`` and numeric elements.

    Duplicate epochs (re-published element sets) keep the most recently created one.
    Output plugs into ``tle_conversion.add_orbital_state``.
    """
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["EPOCH"] = pd.to_datetime(df["EPOCH"], utc=True)
    if "CREATION_DATE" in df:
        df["CREATION_DATE"] = pd.to_datetime(df["CREATION_DATE"], utc=True)
    for col in NUMERIC_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "NORAD_CAT_ID" in df:
        df["NORAD_CAT_ID"] = pd.to_numeric(df["NORAD_CAT_ID"]).astype(int)
    sort_cols = ["EPOCH"] + (["CREATION_DATE"] if "CREATION_DATE" in df else [])
    df = df.sort_values(sort_cols).drop_duplicates(subset=["NORAD_CAT_ID", "EPOCH"]
                                                   if "NORAD_CAT_ID" in df else ["EPOCH"],
                                                   keep="last")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------- tle.toml
TLE_FIELDS = [("epoch_utc", "EPOCH"), ("line1", "TLE_LINE1"), ("line2", "TLE_LINE2"),
              ("inclination_deg", "INCLINATION"), ("raan_deg", "RA_OF_ASC_NODE"),
              ("eccentricity", "ECCENTRICITY"), ("arg_perigee_deg", "ARG_OF_PERICENTER"),
              ("mean_anomaly_deg", "MEAN_ANOMALY"), ("mean_motion_rev_day", "MEAN_MOTION"),
              ("bstar", "BSTAR")]


def _toml_value(v) -> str:
    if isinstance(v, str):
        return json.dumps(v)                         # valid TOML basic string
    if isinstance(v, (pd.Timestamp, datetime)):
        return json.dumps(pd.Timestamp(v).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    return repr(float(v))


def write_tle_toml(gp: pd.DataFrame, key_by_norad: dict[int, str], path: str | Path,
                   retrieved_utc: str, source: str = "Space-Track.org GP class") -> Path:
    """Write the latest elements per satellite (keys from satellites.toml)."""
    lines = ["# Latest orbital elements per satellite. GENERATED -- do not edit or commit.",
             f"# Source: {source}. Retrieved {retrieved_utc}.",
             "# Regenerate with:  python tools/fetch_tles.py", ""]
    latest = gp.sort_values("EPOCH").groupby("NORAD_CAT_ID").tail(1)
    for row in latest.sort_values("NORAD_CAT_ID").itertuples(index=False):
        rec = row._asdict()
        key = key_by_norad.get(int(rec["NORAD_CAT_ID"]), str(rec["NORAD_CAT_ID"]))
        lines.append(f"[{key}]")
        lines.append(f"norad_id = {int(rec['NORAD_CAT_ID'])}")
        for name, col in TLE_FIELDS:
            lines.append(f"{name} = {_toml_value(rec[col])}")
        lines.append(f"source = {json.dumps(source)}")
        lines.append(f"retrieved_utc = {json.dumps(retrieved_utc)}")
        lines.append("")
    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_tle_toml(path: str | Path) -> dict[str, dict]:
    import tomllib
    with open(path, "rb") as fh:
        return tomllib.load(fh)
