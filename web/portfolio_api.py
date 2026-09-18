"""Separate, non-production API for the offline score/whitelist workflow."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, StringConstraints

from web.portfolio_service import PortfolioError, PortfolioService


@lru_cache(maxsize=1)
def get_portfolio_service() -> PortfolioService:
    try:
        return PortfolioService()
    except Exception as exc:
        raise HTTPException(500, "离线配置存储暂时不可用") from exc


def _guard_mutation(request: Request) -> None:
    if request.method != "POST":
        return
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "请使用 application/json")
    try:
        too_large = int(request.headers.get("content-length", "0")) > 512 * 1024
    except ValueError:
        too_large = True
    if too_large:
        raise HTTPException(413, "离线配置请求过大")
    origin = request.headers.get("origin")
    if origin:
        try:
            source, target = urlsplit(origin), urlsplit(str(request.url))
            allowed_hosts = {"localhost", "127.0.0.1", "::1"}
            same_origin = (source.scheme == target.scheme and source.hostname == target.hostname
                           and (source.port or (443 if source.scheme == "https" else 80))
                           == (target.port or (443 if target.scheme == "https" else 80)))
            valid = same_origin and source.hostname in allowed_hosts and not source.username and not source.password
            valid = valid and not source.path and not source.query and not source.fragment
        except ValueError:
            valid = False
        if not valid:
            raise HTTPException(403, "仅允许本机同源页面提交离线配置")


router = APIRouter(prefix="/api/v2/offline-portfolio", tags=["offline-portfolio"],
                   dependencies=[Depends(_guard_mutation)])


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Code = Annotated[StrictStr, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")]
Industry = Annotated[StrictStr, StringConstraints(min_length=1, max_length=100)]


class BundleImportRequest(StrictRequest):
    manifest_path: StrictStr = Field(min_length=1, max_length=1000)


class WhitelistRequest(StrictRequest):
    snapshot_id: StrictStr = Field(min_length=1, max_length=160)
    name: StrictStr = Field(min_length=1, max_length=100)
    codes: list[Code] = Field(max_length=300)
    whitelist_id: StrictStr | None = Field(default=None, max_length=160)
    expected_version: int | None = Field(default=None, ge=1, strict=True)


class Holding(StrictRequest):
    code: Code
    current_weight: float = Field(ge=0, le=1, strict=True)
    industry: StrictStr | None = Field(default=None, max_length=100)
    can_buy: StrictBool | None = None
    can_sell: StrictBool | None = None
    quantity: int | None = Field(default=None, ge=0, strict=True)


class Tradability(StrictRequest):
    can_buy: StrictBool | None = None
    can_sell: StrictBool | None = None


class CostModel(StrictRequest):
    commission_rate: float = Field(default=.00025, ge=0, le=.1, strict=True)
    transfer_rate: float = Field(default=.00001, ge=0, le=.1, strict=True)
    buy_tax_rate: float = Field(default=0., ge=0, le=.1, strict=True)
    sell_tax_rate: float = Field(default=.0005, ge=0, le=.1, strict=True)
    slippage_rate: float = Field(default=.0005, ge=0, le=.1, strict=True)
    minimum_commission: float = Field(default=0., ge=0, le=10000, strict=True)


class PlanRequest(StrictRequest):
    range_utility_tolerance_bps: float = Field(default=1., ge=0, le=100, strict=True)
    whitelist_id: StrictStr = Field(min_length=1, max_length=160)
    whitelist_version: int = Field(ge=1, strict=True)
    market_budget: float = Field(default=1.0, ge=0, le=1, strict=True)
    holdings: list[Holding] | None = Field(default=None, max_length=1000)
    industries: dict[Code, Industry] = Field(default_factory=dict, max_length=1300)
    account_value: float | None = Field(default=None, ge=1e-8, le=1e16, strict=True)
    buy_cost: float = Field(default=0.00076, ge=0, le=0.1, strict=True)
    sell_cost: float = Field(default=0.00126, ge=0, le=0.1, strict=True)
    assume_tradable: StrictBool = False
    tradability: dict[Code, Tradability] = Field(default_factory=dict, max_length=1300)
    minimum_weight: float | None = Field(default=None, ge=.000001, le=.05, strict=True)
    analysis_id: StrictStr | None = Field(default=None, max_length=160)
    volatility_cap: float = Field(default=.12, ge=.001, le=1, strict=True)
    risk_aversion: float = Field(default=3., ge=.01, le=100, strict=True)
    no_trade_band: float = Field(default=.005, ge=0, le=.05, strict=True)
    cost_model: CostModel | None = None


class AnalysisImportRequest(BundleImportRequest):
    snapshot_id: StrictStr = Field(min_length=1, max_length=160)


class ForwardEventRequest(StrictRequest):
    kind: Literal["adopted", "declined", "execution", "valuation", "note"]
    observed_at: StrictStr = Field(min_length=10, max_length=50)
    account_equity: float | None = Field(default=None, gt=0, le=1e16, strict=True)
    execution_status: dict[Code, Literal["simulated_filled", "blocked_buy", "blocked_sell", "pending"]] = Field(default_factory=dict, max_length=1300)
    note: StrictStr = Field(default="", max_length=2000)


def _call(method, *args, **kwargs) -> Any:
    from strategy_manager.portfolio_models import AllocationError
    try:
        return method(*args, **kwargs)
    except PortfolioError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except AllocationError as exc:
        raise HTTPException(422, {"code": exc.code, "message": str(exc), "details": exc.details}) from exc
    except Exception as exc:
        raise HTTPException(500, "离线配置处理失败；原始评分包未修改") from exc


@router.get("/bundles")
def list_bundles(service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.list_bundles)


@router.post("/bundles/import")
def import_bundle(req: BundleImportRequest, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.import_bundle, req.manifest_path)


@router.get("/bundles/{bundle_id}")
def get_bundle(bundle_id: str, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.get_bundle, bundle_id)


@router.get("/scores")
def get_scores(bundle_id: str, date: str, fold_id: str,
               service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.get_scores, bundle_id, date, fold_id)


@router.get("/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.get_snapshot, snapshot_id)


@router.post("/whitelists")
def save_whitelist(req: WhitelistRequest, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.save_whitelist, **req.model_dump())


@router.get("/whitelists")
def list_whitelists(service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.list_whitelists)


@router.get("/whitelists/{whitelist_id}")
def get_whitelist(whitelist_id: str, version: int | None = Query(default=None, ge=1),
                  service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.get_whitelist, whitelist_id, version)


@router.post("/plans")
def create_plan(req: PlanRequest, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.create_plan, req.model_dump())


@router.get("/plans/{plan_id}")
def get_plan(plan_id: str, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.get_plan, plan_id)


@router.get("/analysis")
def get_analysis(snapshot_id: str, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.get_analysis, snapshot_id)


@router.get("/validation")
def get_validation(snapshot_id: str, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.list_validation_reports, snapshot_id)


@router.post("/analysis/import")
def import_analysis(req: AnalysisImportRequest, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.import_analysis, req.manifest_path, req.snapshot_id)


@router.get("/plans/{plan_id}/events")
def get_events(plan_id: str, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.list_forward_events, plan_id)


@router.post("/plans/{plan_id}/events")
def add_event(plan_id: str, req: ForwardEventRequest, service: PortfolioService = Depends(get_portfolio_service)) -> dict:
    return _call(service.record_forward_event, plan_id, req.model_dump())
