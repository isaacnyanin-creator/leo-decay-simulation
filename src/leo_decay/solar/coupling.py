"""Run the unchanged RK45 propagator on time-varying generated forcing.

The propagator sees the forcing as linear interpolation between grid nodes,
so the right-hand side has a kink at every node (storm onsets, flares, and the
red noise in the background all put them there). RK45 at rtol 1e-9 takes
multi-day steps and its error estimate can't see kinks inside a step: daily
segments were off by 0.9 m (30-day average input) to 23 m (instantaneous) over
two years. ``propagate_scenario`` therefore splits integration at every grid
node, where the forcing is exactly linear inside each segment, calling
``propagator.propagate`` per segment and carrying the state across.
"""

from __future__ import annotations

import numpy as np

from ..constants import R_EARTH
from ..propagator import Trajectory, propagate
from .generator import SolarScenario

DEFAULT_MAX_SEGMENT_S = 86_400.0


def _interp_fn(t_nodes, values, offset):
    t_nodes = np.asarray(t_nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    return lambda t: float(np.interp(t + offset, t_nodes, values))


def propagate_piecewise(h0_m: float, duration_s: float, b_star: float,
                        t_s, f107, ap, breakpoints_s=(),
                        max_segment_s: float = DEFAULT_MAX_SEGMENT_S,
                        t_eval_s=None) -> Trajectory:
    """Integrate Eq. B.3 with linearly interpolated ``f107(t_s)``, ``ap(t_s)``.

    Returns samples at ``t_eval_s`` (default: the segment edges).
    """
    duration_s = float(duration_s)
    regular = np.arange(0.0, duration_s, max_segment_s)
    bp = np.asarray(breakpoints_s, dtype=float)
    edges = np.unique(np.concatenate([regular, bp[(bp > 0) & (bp < duration_s)], [duration_s]]))
    edges = np.insert(edges[edges > 0], 0, 0.0)

    t_eval = edges if t_eval_s is None else np.unique(np.asarray(t_eval_s, dtype=float))
    if t_eval.size and (t_eval[0] < 0 or t_eval[-1] > duration_s):
        raise ValueError("t_eval_s must lie within [0, duration_s]")

    out_t, out_a = [], []
    h = float(h0_m)
    for i, (t0, t1) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        req = t_eval[(t_eval >= t0) & ((t_eval <= t1) if last else (t_eval < t1))] - t0
        dur = t1 - t0
        # Always sample the segment start: if re-entry happens before the first
        # requested point, solve_ivp would otherwise return no samples at all.
        local = np.union1d(req, [0.0, dur])
        traj = propagate(h, dur, b_star, _interp_fn(t_s, f107, t0), _interp_fn(t_s, ap, t0),
                         t_eval_s=local)
        keep = np.isin(local[:traj.t_s.size], req)
        out_t.append(traj.t_s[keep] + t0)
        out_a.append(traj.a_m[keep])
        if traj.reentered:
            return Trajectory(t_s=np.concatenate(out_t), a_m=np.concatenate(out_a), reentered=True)
        h = float(traj.a_m[-1] - R_EARTH)

    return Trajectory(t_s=np.concatenate(out_t), a_m=np.concatenate(out_a), reentered=False)


def propagate_scenario(scenario: SolarScenario, h0_m: float, b_star: float,
                       f107_mode: str = "30ma", t_eval_s=None,
                       max_segment_s: float | None = None) -> Trajectory:
    """Propagate through a generated scenario. ``f107_mode``: '30ma' (paper,
    matches the B* calibration) or 'instantaneous' (outside the calibration).

    ``max_segment_s`` defaults to the grid step, so every interpolation node is
    a segment boundary.
    """
    (t, f), (_, ap) = scenario.forcing(f107_mode)
    return propagate_piecewise(h0_m, scenario.duration_s, b_star, t, f, ap,
                               max_segment_s=max_segment_s or scenario.step_s,
                               t_eval_s=t_eval_s)
