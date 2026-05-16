"""
FastAPI application for the SHL Assessment Recommender.

Endpoints:
  GET  /        → {"message": "..."}
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent import process_chat
from .catalog import catalog
from .retrieval import retriever
from .schemas import ChatRequest, ChatResponse, HealthResponse

# Load environment variables from .env
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load catalog and build search index."""
    catalog.load()
    retriever.build_index()
    print(f"[main] SHL Assessment Recommender ready — {len(catalog.items)} assessments indexed")
    yield


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent that recommends SHL assessments based on hiring needs.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend and any origin for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Welcome endpoint."""
    return {"message": "SHL Assessment Recommender API is running!"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a conversational chat request.

    Accepts the full stateless conversation history and returns
    a reply with optional assessment recommendations.
    """
    return await process_chat(request)


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
