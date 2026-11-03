"""Compare the paper's Protocol 3 forecast with observed Hubble TLEs (cached GP_HISTORY).

    python tools/validate_protocol3.py

Reads data/raw/spacetrack/gp_history_20580.json only; makes no network request.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_decay.spacetrack import SpaceTrack, to_dataframe  # noqa: E402
from leo_decay.validation import check_forecast  # noqa: E402

T0 = pd.Timestamp("2026-10-15T00:00:00Z")      # paper gives the date only; 00:00 UTC assumed
PAPER_KM = {7: 469.0302, 14: 468.5559, 21: 468.0771, 28: 467.5936, 35: 467.1053}


def main() -> int:
    entry = SpaceTrack().load_cached_history(20580)
    gp = to_dataframe(entry.records)
    res = check_forecast(gp, T0, 469_500.0, 1.143720e-2, 145.0, 15.0, list(PAPER_KM), PAPER_KM)

    print(f"Hubble GP_HISTORY: {len(gp)} element sets, retrieved {entry.meta['retrieved_utc']}; "
          f"{res.n_tles} since {T0:%Y-%m-%d}, last epoch {gp['EPOCH'].iloc[-1]:%Y-%m-%d %H:%M} UTC")
    print(f"Start {T0:%Y-%m-%d %H:%M} UTC: paper h0 = {res.h0_paper_m / 1e3:.4f} km, "
          f"observed = {res.h0_observed_m / 1e3:.4f} km "
          f"(paper - observed = {res.h0_paper_m - res.h0_observed_m:+.1f} m)")
    cols = ["checkpoint_utc", "h_paper_m", "h_pred_m", "h_obs_interp_m", "residual_m",
            "residual_nearest_m", "residual_anchored_m", "bracket_gap_h"]
    with pd.option_context("display.float_format", "{:,.1f}".format, "display.width", 200):
        print(res.table[cols].to_string(index=False))
    print(f"Mean decay rate since start: observed {res.observed_rate_km_day:+.4f} km/day, "
          f"forecast {res.forecast_rate_km_day:+.4f} km/day")
    print(f"TLE altitude noise (std about 30-day quadratic): {res.noise_m:.1f} m")

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    res.table.to_csv(out / "protocol3_residuals.csv", index=False)
    print(f"Saved {(out / 'protocol3_residuals.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
