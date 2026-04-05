"""TurbulenceSim integration package.

This package provides a local adapter wrapper that can be replaced by the
upstream TurbulenceSim-v1 code from Purdue.
"""

from .simulator import TurbulenceSimV1Adapter

__all__ = ["TurbulenceSimV1Adapter"]
