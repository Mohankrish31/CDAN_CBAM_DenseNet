# __init__.py
"""
CDAN_CBAM_DenseNet Package
--------------------------
This package provides the CDAN + CBAM + DenseNet model for low-light image enhancement.
It exposes the main model class for easy import.
"""
from .model import CDAN_CBAM_DenseNet  # Import the main model class
__all__ = ['CDAN_CBAM_DenseNet']       # Define what is available for import
