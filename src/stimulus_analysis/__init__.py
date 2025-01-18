"""
Stimulus analysis module for Drosophila behavior analysis.

This module provides tools for analyzing fly responses to visual stimuli,
including heading analysis and behavioral categorization.
"""

from .heading_analyzer import HeadingAnalyzer, TimeWindow
from .binned_analyzer import (
    BinnedHeadingAnalyzer,
    BinConfig,
    BinnedResult,
    VelocityResult
)

__all__ = [
    'HeadingAnalyzer',
    'TimeWindow',
    'BinnedHeadingAnalyzer',
    'BinConfig',
    'BinnedResult',
    'VelocityResult'
]

# Version of the stimulus_analysis module
__version__ = '0.1.0'

# Default time windows for analysis
DEFAULT_BEFORE_WINDOW = (495, 535)
DEFAULT_AFTER_WINDOW = (620, 660)
DEFAULT_FPS = 25
