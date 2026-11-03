"""Download TLEs from Space-Track (policy-aware, cached) and regenerate config/tle.toml.

    python tools/fetch_tles.py                 # latest GP for all config satellites
    python tools/fetch_tles.py --history 20580 2026-01-01 2026-11-03

Credentials come from .env (SPACETRACK_USER / SPACETRACK_PASSWORD) and are never printed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_decay.config import load_satellites  # noqa: E402
from leo_decay.spacetrack import (  # noqa: E402
    PolicyViolation, SpaceTrack, credentials_available, to_dataframe, write_tle_toml,
)

TLE_TOML = ROOT / "config" / "tle.toml"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--history", nargs=3, metavar=("NORAD", "START", "END"),
                    help="one-time GP_HISTORY download for one object (cached permanently)")
    ap.add_argument("--skip-gp", action="store_true", help="don't request latest GP elements")
    args = ap.parse_args(argv)

    status = credentials_available()
    if not all(status.values()):
        print("Credentials not set:", ", ".join(k for k, v in status.items() if not v))
        return 2

    sats = load_satellites()
    key_by_norad = {s.norad_id: k for k, s in sats.items()}
    client = SpaceTrack()

    if args.history:
        norad, start, end = int(args.history[0]), date.fromisoformat(args.history[1]), \
            date.fromisoformat(args.history[2])
        entry = client.fetch_gp_history(norad, start, end)
        print(f"GP_HISTORY {norad}: {len(entry.records)} records -> {entry.path.relative_to(ROOT)} "
              f"(retrieved {entry.meta['retrieved_utc']})")

    if not args.skip_gp:
        try:
            entry = client.fetch_latest_gp(key_by_norad)
        except PolicyViolation as exc:
            print(f"GP not requested: {exc}")
            entry = client.latest_cached_gp()
            if entry is None:
                return 1
            print(f"Using cached {entry.path.name}")
        gp = to_dataframe(entry.records)
        write_tle_toml(gp, key_by_norad, TLE_TOML, entry.meta["retrieved_utc"])
        for row in gp.itertuples():
            print(f"  {row.NORAD_CAT_ID:>6} {row.OBJECT_NAME:<24} epoch {row.EPOCH:%Y-%m-%d %H:%M} UTC")
        print(f"Wrote {TLE_TOML.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
