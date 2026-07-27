"""Provider routing: which model serves which workload, and what it costs."""

from .base import ModelSpec, Selection, Workload
from .registry import ROUTING_TABLE, all_specs, spec_by_id
from .router import Router

__all__ = [
    "ModelSpec",
    "ROUTING_TABLE",
    "Router",
    "Selection",
    "Workload",
    "all_specs",
    "spec_by_id",
]
