# main.py

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from src.video_analysis.video_processor import VideoProcessor
from src.saccade_detection.detector import SaccadeDetector, SaccadeParameters
from src.data_processing.trajectory_filter import TrajectoryFilter
from src.data_processing.bin_segmenter import BinSegmenter
from src.stimulus_analysis.heading_analyzer import HeadingAnalyzer
from src.stimulus_analysis.bin_visualizer import BinComparisonVisualizer


@dataclass
class PipelineConfig:
    """Configuration for the complete analysis pipeline."""
    video_dir: str
    output_dir: str
    calibration_file: str
    before_window: tuple = (495, 535)
    after_window: tuple = (620, 660)
    saccade_threshold: float = 50.0


class AnalysisPipeline:
    """
    Complete pipeline for fly behavior analysis.
    
    Coordinates all analysis steps from video processing
    to final visualization.
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary output directories."""
        dirs = [
            'tracking',
            'saccades',
            'filtered',
            'binned',
            'analysis',
            'visualizations'
        ]
        
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
    def process_videos(self):
        """Process videos to extract tracking data."""
        processor = VideoProcessor()
        tracking_dir = self.output_dir / 'tracking'
        
        return processor.process_folder(
            self.config.video_dir,
            str(tracking_dir)
        )
        
    def detect_saccades(self):
        """Detect saccades in tracking data."""
        params = SaccadeParameters(height=self.config.saccade_threshold)
        detector = SaccadeDetector(params)
        
        tracking_dir = self.output_dir / 'tracking'
        saccade_dir = self.output_dir / 'saccades'
        
        return detector.process_files(
            str(tracking_dir),
            str(saccade_dir)
        )
        
    def filter_trajectories(self):
        """Filter valid trajectory data."""
        trajectory_filter = TrajectoryFilter()
        saccade_dir = self.output_dir / 'saccades'
        filtered_dir = self.output_dir / 'filtered'
        
        return trajectory_filter.filter_trajectories(
            str(saccade_dir),
            str(filtered_dir)
        )
        
    def analyze_headings(self):
        """Analyze heading changes."""
        analyzer = HeadingAnalyzer(self.config.calibration_file)
        filtered_dir = self.output_dir / 'filtered'
        analysis_dir = self.output_dir / 'analysis'
        
        before_results = analyzer.analyze_folder(
            str(filtered_dir),
            str(analysis_dir / 'before'),
            window=self.config.before_window
        )
        
        after_results = analyzer.analyze_folder(
            str(filtered_dir),
            str(analysis_dir / 'after'),
            window=self.config.after_window
        )
        
        return before_results, after_results
        
    def create_visualizations(self):
        """Generate analysis visualizations."""
        visualizer = BinComparisonVisualizer()
        analysis_dir = self.output_dir / 'analysis'
        viz_dir = self.output_dir / 'visualizations'
        
        for bin_file in (analysis_dir / 'binned').glob('*.csv'):
            output_file = viz_dir / f"{bin_file.stem}_comparison.png"
            visualizer.visualize_comparison(
                str(bin_file),
                str(output_file),
                show_plot=False
            )
            
    def run(self):
        """Execute complete analysis pipeline."""
        print("Starting analysis pipeline...")
        
        print("\n1. Processing videos...")
        self.process_videos()
        
        print("\n2. Detecting saccades...")
        self.detect_saccades()
        
        print("\n3. Filtering trajectories...")
        self.filter_trajectories()
        
        print("\n4. Analyzing headings...")
        self.analyze_headings()
        
        print("\n5. Creating visualizations...")
        self.create_visualizations()
        
        print("\nAnalysis complete!")


def main():
    """Command line interface for the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run complete fly behavior analysis pipeline"
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing input videos"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for analysis outputs"
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="Path to calibration file"
    )
    parser.add_argument(
        "--saccade-threshold",
        type=float,
        default=50.0,
        help="Threshold for saccade detection"
    )
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        calibration_file=args.calibration,
        saccade_threshold=args.saccade_threshold
    )
    
    pipeline = AnalysisPipeline(config)
    
    try:
        pipeline.run()
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
