from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.agent import router as agent_router
from .api.articles import router as articles_router
from .api.authors import router as authors_router
from .api.dashboard import router as dashboard_router
from .api.search import router as search_router
from .openaire.client import close_openaire
from .openalex.client import close_client
from .settings import settings
from .storage.db import init_db
from .taxonomy import Taxonomy, load_taxonomy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("webapp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    log.info("Loading taxonomy (cache=%s)", settings.taxonomy_cache)
    tax = await load_taxonomy()
    app.state.taxonomy = tax
    log.info(
        "Taxonomy ready: %d domains, %d topics",
        len(tax.domains),
        sum(len(s.topics) for d in tax.domains for f in d.fields for s in f.subfields),
    )
    try:
        yield
    finally:
        await close_client()
        await close_openaire()


app = FastAPI(title="Nauka-Monitor API", version="0.0.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(dashboard_router)
app.include_router(articles_router)
app.include_router(authors_router)
app.include_router(agent_router)
app.include_router(search_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/taxonomy")
async def api_taxonomy() -> dict:
    tax: Taxonomy = app.state.taxonomy
    return tax.to_dict()
