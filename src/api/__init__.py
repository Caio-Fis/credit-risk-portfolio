"""Production FastAPI service for the v2 PD pipeline.

Exposes calibrated default probability, SHAP explanations, drift status,
and a recalibration trigger. The frontend (Streamlit) consumes this same
API in production.
"""

from src.api.main import create_app

__all__ = ["create_app"]
