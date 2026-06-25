"""DCRNet v29ng: v29 with nonlinearity=False (no GELU after ComplexLowRankEncoder).

Tests GPT Pro's concern that GELU applied separately to Re and Im of
the complex bilinear measurements breaks phase preservation. With
nonlinearity=False, the bilinear path is fully linear in H, and the
mix Linear handles the rank → codeword projection without distortion.

Everything else identical to v29.
"""
from __future__ import annotations

from .dcrnet_v29 import dcrnet_v29


__all__ = ["dcrnet_v29ng"]


def dcrnet_v29ng(**kwargs):
    kwargs["nonlinearity"] = False
    return dcrnet_v29(**kwargs)
