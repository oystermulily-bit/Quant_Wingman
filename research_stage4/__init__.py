"""Stage-4 Development OOF feasibility gate.

Does not start RD-Agent, does not read Holdout prices or performance, and does
not change pre-registered GO thresholds after seeing results.
"""

from .gate import Stage4Thresholds, evaluate_horizon_go
from .stats import holm_adjust, moving_block_bootstrap, performance_from_returns

__all__ = [
    "Stage4Thresholds",
    "evaluate_horizon_go",
    "holm_adjust",
    "moving_block_bootstrap",
    "performance_from_returns",
]
