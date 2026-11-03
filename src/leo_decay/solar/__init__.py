"""Stochastic solar event generator (placeholder-calibrated).

Produces F10.7(t) and Ap(t) series plus an event log, and feeds them to the
unchanged ``leo_decay.propagator``.
"""

from .config import SolarConfig, load_solar_config
from .generator import SolarScenario, generate_scenario
from .coupling import propagate_piecewise, propagate_scenario

__all__ = [
    "SolarConfig", "load_solar_config",
    "SolarScenario", "generate_scenario",
    "propagate_piecewise", "propagate_scenario",
]
