"""RK45 integration of the explicit expanded SI decay ODE, Eq. B.3 (Appendix A.3).

da/dt = -B* rho(a, F10.7(t), Ap(t)) sqrt(mu a)

Integrated with scipy's Dormand-Prince RK45 at rtol=1e-9, atol=1e-12, with a
terminal event at a <= R_E (re-entry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .constants import MU_EARTH, R_EARTH, SECONDS_PER_DAY
from .density import thermospheric_density

RTOL = 1e-9
ATOL = 1e-12

Forcing = float | Callable[[float], float] | tuple[np.ndarray, np.ndarray]


def make_forcing(value: Forcing) -> Callable[[float], float]:
    """Normalise a forcing input to a function of time (s).

    Accepts a constant, a callable ``f(t_s)``, or a ``(t_s, values)`` pair which
    is linearly interpolated with ``np.interp`` (as in Protocol 2).
    """
    if callable(value):
        return value
    if isinstance(value, tuple):
        t_nodes, v_nodes = (np.asarray(x, dtype=float) for x in value)
        return lambda t: float(np.interp(t, t_nodes, v_nodes))
    const = float(value)
    return lambda t: const


def decay_rhs(t_s: float, y, b_star: float,
              f107_fn: Callable[[float], float],
              ap_fn: Callable[[float], float]):
    """ODE right-hand side; y = [a_m]. Returns [da/dt in m/s]."""
    a_m = y[0]
    if a_m <= R_EARTH:
        return [0.0]
    rho = thermospheric_density(a_m, f107_fn(t_s), ap_fn(t_s))
    return [-b_star * rho * np.sqrt(MU_EARTH * a_m)]


def _reentry_event(t_s, y, *args):
    return y[0] - R_EARTH


_reentry_event.terminal = True
_reentry_event.direction = -1


@dataclass(frozen=True)
class Trajectory:
    t_s: np.ndarray             # s since t0
    a_m: np.ndarray             # semi-major axis, m
    reentered: bool

    @property
    def h_m(self) -> np.ndarray:
        return self.a_m - R_EARTH

    @property
    def t_days(self) -> np.ndarray:
        return self.t_s / SECONDS_PER_DAY


def propagate(h0_m: float, duration_s: float, b_star: float,
              f107: Forcing, ap: Forcing,
              t_eval_s=None,
              rtol: float = RTOL, atol: float = ATOL) -> Trajectory:
    """Integrate Eq. B.3 from altitude ``h0_m`` for ``duration_s`` seconds."""
    sol = solve_ivp(
        decay_rhs, t_span=(0.0, float(duration_s)), y0=[R_EARTH + float(h0_m)],
        method="RK45", t_eval=t_eval_s,
        args=(float(b_star), make_forcing(f107), make_forcing(ap)),
        events=_reentry_event, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"RK45 integration failed: {sol.message}")
    return Trajectory(t_s=sol.t, a_m=sol.y[0], reentered=sol.status == 1)


def weekly_checkpoints_s(n_weeks: int) -> np.ndarray:
    """Checkpoint times t_k = 7k days for k = 0..n_weeks, in seconds."""
    return np.arange(n_weeks + 1) * 7.0 * SECONDS_PER_DAY
