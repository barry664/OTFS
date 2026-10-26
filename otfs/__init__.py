"""OTFS: optimal-transport frame selection for long videos."""

from .core import OTFSConfig, OTFSResult, coverage_dp, otfs_select, semi_unbalanced_sinkhorn

__all__ = [
    "OTFSConfig",
    "OTFSResult",
    "coverage_dp",
    "otfs_select",
    "semi_unbalanced_sinkhorn",
]

__version__ = "0.1.0"
