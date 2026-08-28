from datetime import date
from inspect import signature

import pandas as pd

from data_pipeline.hs300.config import REQUESTED_START, RESEARCH_START
from data_pipeline.hs300_panel import HS300PanelDataManager
from research_stage3.runner import Stage3ResearchRunner


def test_formal_research_start_is_frozen_to_2014() -> None:
    assert RESEARCH_START == date(2014, 1, 2)
    assert REQUESTED_START == date(2010, 1, 1)
    assert REQUESTED_START < RESEARCH_START


def test_panel_and_stage3_defaults_use_formal_research_start() -> None:
    manager = HS300PanelDataManager("unused-for-construction")
    assert manager.required_start == pd.Timestamp(RESEARCH_START)
    assert (
        signature(Stage3ResearchRunner.run)
        .parameters["required_start"]
        .default
        == RESEARCH_START.isoformat()
    )
