"""Space-Track.org TLE download -- STUB.

Credentials are read from the environment (never commit them):

    SPACETRACK_USER, SPACETRACK_PASS

To implement, query the ``gp_history`` class for a NORAD ID and epoch range and
return a DataFrame with at least ``EPOCH`` (datetime) and ``MEAN_MOTION``
(rev/day) columns, so it plugs into ``tle_conversion.add_orbital_state``.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd


def get_credentials() -> tuple[str, str]:
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")
    if not user or not password:
        raise RuntimeError("Set SPACETRACK_USER and SPACETRACK_PASS environment variables.")
    return user, password


def fetch_tles(norad_id: int, start: date, end: date) -> pd.DataFrame:
    """Download historical TLEs for ``norad_id`` between ``start`` and ``end``."""
    raise NotImplementedError(
        "Space-Track download not implemented yet; add credentials and a client."
    )
