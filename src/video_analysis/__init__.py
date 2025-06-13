"""
Video analysis module for fly tracking and behavior analysis.

This module provides tools for processing video data of flying Drosophila,
including blob detection and trajectory extraction.
"""

from .video_processor import VideoProcessor

__all__ = ['VideoProcessor']

# Version of the video_analysis module
__version__ = '0.1.0'
