"""RD-Agent factor-research loop for quant_w1ngman."""

from .research_loop import FactorResearchLoop, RoundOutcome
from .group_search import GroupSearchConfig, GroupSearchResult, StepwiseGroupSearch
from .group_evaluator import FixedLightGBMGroupEvaluator, SelectionFold
from .joint_runner import JointPanel, JointResearchRunner, OuterFold
from .joint_protocol import FrozenJointPlan, RunLedger

__all__ = [
    "FactorResearchLoop", "RoundOutcome", "GroupSearchConfig", "GroupSearchResult",
    "StepwiseGroupSearch", "FixedLightGBMGroupEvaluator", "SelectionFold",
    "JointPanel", "JointResearchRunner", "OuterFold", "FrozenJointPlan", "RunLedger",
]
