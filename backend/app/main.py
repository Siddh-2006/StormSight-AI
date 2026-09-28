from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_alerts import router as alerts_router
from app.api.routes_nowcast import router as nowcast_router

app = FastAPI(
    title="Thunderstorm & Lightning Nowcasting API",
    description="SIH PS 26072 prototype backend — see docs/technical_flow.md for the full contract.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(nowcast_router)
app.include_router(alerts_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "docs": "/docs", "contract": "docs/technical_flow.md"}
