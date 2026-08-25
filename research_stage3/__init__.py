"""Stage 3 point-in-time cross-sectional research loop.

The package does not call RD-Agent, inspect the final holdout, or expose a
production recommendation API.
"""

from .protocol import (
    DateSplitProtocol,
    FoldSplit,
    SplitPlan,
    Stage3Config,
)

__all__ = ["DateSplitProtocol", "FoldSplit", "SplitPlan", "Stage3Config"]
