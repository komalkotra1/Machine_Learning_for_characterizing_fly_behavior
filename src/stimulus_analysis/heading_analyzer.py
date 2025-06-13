# src/stimulus_analysis/heading_analyzer.py

import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple


@dataclass
class TimeWindow:
    """Configuration for analysis time windows."""
    before_stimulus: tuple[int, int] = (495, 535)
    after_stimulus: tuple[int, int] = (620, 660)
    fps: int = 25

    @property
    def frame_duration(self) -> float:
        return 1/self.fps


@dataclass
class AnalysisResults:
    """Container for analysis results."""
    file_name: str
    camera_number: int
    frame_number: int
    heading: float
    interp_heading: float
    difference_in_heading: float


class HeadingAnalyzer:
    """
    Analyzes fly heading before and after stimulus presentation.
    
    Processes trajectory data to compute heading changes in response to stimuli.
    """
    
    def __init__(self, 
                 calibration_file: str,
                 time_window: Optional[TimeWindow] = None):
        """
        Initialize the analyzer.
        
        Args:
            calibration_file: Path to calibration data
            time_window: Configuration for analysis windows
        """
        self.calibration_df = pd.read_csv(calibration_file)
        self.time_window = time_window or TimeWindow()
        
    def extract_metadata(self, file_name: str) -> tuple[Optional[int], Optional[int]]:
        """Extract camera and frame numbers from filename."""
        camera_match = re.search(r'cam_(\d+)', file_name)
        frame_match = re.search(r'frame_(\d+)', file_name)
        
        if camera_match and frame_match:
            return int(camera_match.group(1)), int(frame_match.group(1))
        return None, None
        
    def calculate_heading(self, 
                         x_values: np.ndarray, 
                         y_values: np.ndarray,
                         window: tuple[int, int]) -> Optional[float]:
        """Calculate heading in specified time window."""
        try:
            if len(x_values) <= window[1] or len(y_values) <= window[1]:
                return None
                
            heading = np.arctan2(
                np.diff(y_values)[window[0]:window[1]],
                np.diff(x_values)[window[0]:window[1]]
            )
            
            if heading.size == 0:
                return None
                
            return np.nanmean(heading)
        except Exception as e:
            print(f"Error calculating heading: {e}")
            return None
            
    def interpolate_stimulus_heading(self, 
                                   camera_number: int,
                                   frame_number: int) -> Optional[float]:
        """Interpolate heading based on calibration data."""
        camera_data = self.calibration_df[
            self.calibration_df['camera'] == camera_number
        ]
        
        if camera_data.empty:
            return None
            
        try:
            heading = camera_data['heading_direction'].values
            screen = camera_data['stim_direction'].values
            
            if heading.size == 0 or screen.size == 0:
                return None
                
            return np.interp(
                frame_number, 
                screen, 
                heading, 
                period=2 * np.pi
            )
        except Exception as e:
            print(f"Interpolation error: {e}")
            return None
            
    def process_file(self, file_path: str) -> Optional[AnalysisResults]:
        """Process a single trajectory file."""
        file_name = Path(file_path).name
        camera_number, frame_number = self.extract_metadata(file_name)
        
        if not (camera_number and frame_number):
            return None
            
        try:
            df = pd.read_csv(file_path)
            x_values = df['x'].values
            y_values = df['y'].values
            
            heading = self.calculate_heading(
                x_values, 
                y_values, 
                self.time_window.before_stimulus
            )
            
            if heading is None:
                return None
                
            interp_heading = self.interpolate_stimulus_heading(
                camera_number,
                frame_number
            )
            
            if interp_heading is None:
                return None
                
            difference = np.arctan2(
                np.sin(heading - interp_heading),
                np.cos(heading - interp_heading)
            )
            
            return AnalysisResults(
                file_name=file_name,
                camera_number=camera_number,
                frame_number=frame_number,
                heading=heading,
                interp_heading=interp_heading,
                difference_in_heading=difference
            )
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            return None
            
    def analyze_folder(self, 
                      input_folder: str, 
                      output_folder: str) -> pd.DataFrame:
        """
        Analyze all trajectory files in a folder.
        
        Args:
            input_folder: Path to trajectory files
            output_folder: Path to save results
            
        Returns:
            DataFrame with analysis results
        """
        results = []
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        for file_path in input_path.glob("*.csv"):
            result = self.process_file(str(file_path))
            if result:
                results.append(vars(result))
                
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            # Save results
            os.makedirs(output_path, exist_ok=True)
            results_df.to_csv(
                output_path / "heading_analysis_results.csv",
                index=False
            )
            
            # Create visualization
            self.plot_results(results_df, output_path)
            
        return results_df
        
    def plot_results(self, 
                    results_df: pd.DataFrame,
                    output_path: Path) -> None:
        """Create visualization of heading differences."""
        plt.figure(figsize=(10, 6))
        
        # Convert to degrees for plotting
        differences = np.rad2deg(results_df['difference_in_heading'].values)
        instances = range(1, len(differences) + 1)
        
        plt.bar(instances, differences)
        plt.xlabel("Fly Instance")
        plt.ylabel("Difference in Heading (Degrees)")
        plt.title("Heading Differences Across All Flies")
        
        plt.savefig(output_path / "heading_differences.png")
        plt.close()


def main():
    """Command line interface for heading analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze fly heading relative to stimulus"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input folder with trajectory files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for results"
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="Path to calibration file"
    )
    parser.add_argument(
        "--before-window",
        nargs=2,
        type=int,
        default=[495, 535],
        help="Frame range before stimulus (start end)"
    )
    parser.add_argument(
        "--after-window",
        nargs=2,
        type=int,
        default=[620, 660],
        help="Frame range after stimulus (start end)"
    )
    
    args = parser.parse_args()
    
    time_window = TimeWindow(
        before_stimulus=tuple(args.before_window),
        after_stimulus=tuple(args.after_window)
    )
    
    analyzer = HeadingAnalyzer(args.calibration, time_window)
    results_df = analyzer.analyze_folder(args.input, args.output)
    print(f"Processed {len(results_df)} files successfully")


if __name__ == "__main__":
    main()
