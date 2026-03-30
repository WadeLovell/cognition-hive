"""
CognitionHive API Server

FastAPI wrapper around the CognitionHive pipeline.
Exposes the agent pipeline as a REST endpoint for the React frontend.
"""

import os
import time
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import load_config, init_agents, process_request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("cognition-hive.api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CognitionHive",
    description="Verification-first refinancing analysis",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cognitionhive.com",
        "https://www.cognitionhive.com",
        "https://cognition-hive-frontend.vercel.app",
        "https://cognition-hive-frontend-wadelovells-projects.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Agent initialization (once at startup)
# ---------------------------------------------------------------------------

config = load_config(Path("config"))
agents = init_agents(config, include_warden=False)
logger.info(f"Agents initialized: {', '.join(agents.keys())}")

# ---------------------------------------------------------------------------
# Rate limiting (in-memory, resets on restart)
# ---------------------------------------------------------------------------

RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "3"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "86400"))

rate_limit_store: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(client_ip: str) -> bool:
    """Return True if the request is allowed, False if rate limited."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Remove expired timestamps
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip] if t > window_start
    ]

    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
        return False

    rate_limit_store[client_ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Response cache (in-memory, simple hash-based)
# ---------------------------------------------------------------------------

CACHE_TTL = int(os.environ.get("CACHE_TTL", "3600"))

response_cache: dict[str, dict] = {}


def get_cache_key(request_text: str) -> str:
    """Normalize and hash the request for cache lookup."""
    normalized = request_text.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def get_cached_response(request_text: str) -> dict | None:
    """Return cached response if available and not expired."""
    key = get_cache_key(request_text)
    if key in response_cache:
        entry = response_cache[key]
        if time.time() - entry["timestamp"] < CACHE_TTL:
            logger.info(f"Cache hit for key {key}")
            return entry["response"]
        else:
            del response_cache[key]
    return None


def cache_response(request_text: str, response: dict) -> None:
    """Store response in cache."""
    key = get_cache_key(request_text)
    response_cache[key] = {
        "response": response,
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    request: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The refinancing question to analyze",
        examples=[
            "Should I refinance my 6.5% mortgage with a $350,000 balance and 25 years remaining?",
            "Compare current 30-year fixed rates from major lenders",
        ],
    )

    # Optional context the user can provide for better analysis
    current_rate: float | None = Field(
        None,
        ge=0,
        le=20,
        description="Current mortgage interest rate as a percentage",
    )
    loan_balance: float | None = Field(
        None,
        ge=0,
        description="Remaining loan balance in dollars",
    )
    remaining_years: int | None = Field(
        None,
        ge=1,
        le=40,
        description="Years remaining on current mortgage",
    )


class AnalysisResponse(BaseModel):
    session_id: str
    status: str
    category: str
    output: str | None
    caveats: list[str]
    claims_found: int
    verification_confidence: float
    verification_recommendation: str
    cached: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agents": list(agents.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(body: AnalysisRequest, request: Request):
    """
    Run a refinancing question through the CognitionHive pipeline.

    Rate limited to 3 requests per day per IP address.
    Responses are cached for 1 hour.
    """
    # Rate limit check
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "message": f"Limited to {RATE_LIMIT_MAX} queries per day. Try again tomorrow.",
                "limit": RATE_LIMIT_MAX,
                "window_seconds": RATE_LIMIT_WINDOW,
            },
        )

    # Build the full request string with optional context
    full_request = body.request
    context_parts = []
    if body.current_rate is not None:
        context_parts.append(f"current rate: {body.current_rate}%")
    if body.loan_balance is not None:
        context_parts.append(f"loan balance: ${body.loan_balance:,.0f}")
    if body.remaining_years is not None:
        context_parts.append(f"remaining term: {body.remaining_years} years")

    if context_parts:
        full_request += f" (My details: {', '.join(context_parts)})"

    # Check cache
    cached = get_cached_response(full_request)
    if cached:
        cached["cached"] = True
        return AnalysisResponse(**cached)

    # Run the pipeline
    try:
        result = process_request(full_request, agents)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "pipeline_error",
                "message": "Analysis failed. Please try again.",
            },
        )

    # Build response
    response_data = {
        "session_id": result.get("session_id", "unknown"),
        "status": result.get("status", "unknown"),
        "category": result.get("_category", "research"),
        "output": result.get("output"),
        "caveats": result.get("caveats", []),
        "claims_found": result.get("_claims_found", 0),
        "verification_confidence": result.get("_verification_confidence", 0.0),
        "verification_recommendation": result.get("_verification_recommendation", "unknown"),
        "cached": False,
    }

    # Cache the response
    cache_response(full_request, response_data)

    return AnalysisResponse(**response_data)


@app.get("/rate-limit-status")
async def rate_limit_status(request: Request):
    """Check how many queries the user has remaining today."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    recent = [
        t for t in rate_limit_store.get(client_ip, []) if t > window_start
    ]

    return {
        "queries_used": len(recent),
        "queries_remaining": max(0, RATE_LIMIT_MAX - len(recent)),
        "limit": RATE_LIMIT_MAX,
        "window_seconds": RATE_LIMIT_WINDOW,
    }
