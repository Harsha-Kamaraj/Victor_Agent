"""The routing table.

Numbers below are the published free-tier allowances as of the build date.
They are declared conservatively on purpose: if a provider is more generous
than this table says, Victor under-uses it and nothing breaks. If the table
were optimistic instead, the agent would hit a 429 mid-task and the "$0"
promise would turn into a stall in front of whoever is watching the demo.

Providers change these without notice. ``victor doctor --network`` checks the
keys work; it cannot check the numbers, so treat them as a floor.
"""

from __future__ import annotations

from ..quota import UNMETERED, QuotaLimits
from .base import ModelSpec, Workload

GROQ_DAY = "UTC"
GOOGLE_DAY = "America/Los_Angeles"  # Google's free tier rolls at midnight Pacific.


# --- text reasoning -------------------------------------------------------

GPT_OSS_120B = ModelSpec(
    provider="groq",
    model="openai/gpt-oss-120b",
    workloads=(Workload.TEXT,),
    credential="groq_api_key",
    limits=QuotaLimits(
        requests_per_minute=30,
        requests_per_day=1_000,
        tokens_per_minute=8_000,
        reset_timezone=GROQ_DAY,
    ),
    notes="Primary reasoner. Native tool calling, fast enough for a ReAct loop.",
)

LLAMA_33_70B = ModelSpec(
    provider="groq",
    model="llama-3.3-70b-versatile",
    workloads=(Workload.TEXT,),
    credential="groq_api_key",
    limits=QuotaLimits(
        requests_per_minute=30,
        requests_per_day=1_000,
        tokens_per_minute=12_000,
        reset_timezone=GROQ_DAY,
    ),
    notes="Text fallback on a separate per-model bucket from gpt-oss-120b.",
)

LLAMA_31_8B = ModelSpec(
    provider="groq",
    model="llama-3.1-8b-instant",
    workloads=(Workload.TEXT,),
    credential="groq_api_key",
    limits=QuotaLimits(
        requests_per_minute=30,
        requests_per_day=14_400,
        tokens_per_minute=6_000,
        reset_timezone=GROQ_DAY,
    ),
    notes="Last-resort text. Weak at tool choice, but its 14.4k/day is the "
    "reason the agent can degrade instead of stopping.",
)


# --- vision ---------------------------------------------------------------

GEMINI_25_FLASH = ModelSpec(
    provider="gemini",
    model="gemini-2.5-flash",
    workloads=(Workload.VISION,),
    credential="gemini_api_key",
    limits=QuotaLimits(
        requests_per_minute=10,
        requests_per_day=250,
        reset_timezone=GOOGLE_DAY,
    ),
    notes="Primary vision. Scarcest resource in the stack - spend it last.",
)

LLAMA_4_SCOUT = ModelSpec(
    provider="groq",
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    workloads=(Workload.VISION, Workload.TEXT),
    credential="groq_api_key",
    limits=QuotaLimits(
        requests_per_minute=30,
        requests_per_day=1_000,
        tokens_per_minute=30_000,
        reset_timezone=GROQ_DAY,
    ),
    notes="Vision fallback once Gemini's 250/day is gone.",
)


# --- speech ---------------------------------------------------------------

WHISPER_TURBO = ModelSpec(
    provider="groq",
    model="whisper-large-v3-turbo",
    workloads=(Workload.STT,),
    credential="groq_api_key",
    limits=QuotaLimits(
        requests_per_minute=20,
        requests_per_day=2_000,
        audio_seconds_per_day=28_800,
        reset_timezone=GROQ_DAY,
    ),
    notes="8 hours of audio a day. Not the binding constraint.",
)

PIPER_LOCAL = ModelSpec(
    provider="piper",
    model="en_US-lessac-medium",
    workloads=(Workload.TTS,),
    limits=UNMETERED,
    local=True,
    notes="ONNX, on-CPU, offline. No quota, no latency floor from the network.",
)


# --- embeddings -----------------------------------------------------------

FASTEMBED_LOCAL = ModelSpec(
    provider="fastembed",
    model="BAAI/bge-small-en-v1.5",
    workloads=(Workload.EMBEDDING,),
    limits=UNMETERED,
    local=True,
    notes="384-dim, ONNX runtime. Backs both P6 memory and P7 scout.",
)


ROUTING_TABLE: dict[Workload, tuple[ModelSpec, ...]] = {
    # Ordered best-first. The router walks these in order and takes the first
    # one that has both a credential and remaining allowance.
    Workload.TEXT: (GPT_OSS_120B, LLAMA_33_70B, LLAMA_31_8B),
    Workload.VISION: (GEMINI_25_FLASH, LLAMA_4_SCOUT),
    Workload.STT: (WHISPER_TURBO,),
    Workload.TTS: (PIPER_LOCAL,),
    Workload.EMBEDDING: (FASTEMBED_LOCAL,),
}

#: Which settings field each workload's chain needs at minimum, for `doctor`.
WORKLOAD_CREDENTIALS: dict[Workload, tuple[str, ...]] = {
    workload: tuple(
        dict.fromkeys(s.credential for s in specs if s.credential is not None)
    )
    for workload, specs in ROUTING_TABLE.items()
}


def all_specs() -> tuple[ModelSpec, ...]:
    """Every distinct model in the table, in routing order."""
    seen: dict[str, ModelSpec] = {}
    for specs in ROUTING_TABLE.values():
        for spec in specs:
            seen.setdefault(spec.key, spec)
    return tuple(seen.values())


def spec_by_id(key: str) -> ModelSpec | None:
    """Look up a spec by ``provider:model``, or by bare model name."""
    for spec in all_specs():
        if key in (spec.key, spec.model):
            return spec
    return None


def limits_by_key() -> dict[str, QuotaLimits]:
    """``{quota key: limits}`` for every model - what the ledger displays."""
    return {spec.key: spec.limits for spec in all_specs()}
