"""Small local app for the offline portfolio workspace; no training startup."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.portfolio_api import router
from web.v2_status import refuse_recommendation

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="W1ngman offline portfolio", version="1.0.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
@app.get("/portfolio", include_in_schema=False)
def portfolio_page():
    return FileResponse(STATIC / "portfolio.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "offline_portfolio", "production_allowed": False}


@app.post("/api/v2/portfolio/recommendation")
def production_recommendation():
    return refuse_recommendation()
