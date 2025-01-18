# src/stimulus_analysis/binned_analyzer.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from .heading_analyzer import TimeWindow

@dataclass
class BinConfig:
    """Configuration for heading angle binning."""
    bin_edges: List[float] = None
    use_absolute_values: bool = False
    
    def __post_init__(self):
        if self.bin_edges is None:
            self.bin_edges = [0, 45, 90, 135, 180] if self.use_absolute_values else \
                           [-180, -135, -90, -45, 0, 45, 90, 135, 180]

@dataclass
class VelocityResult:
    """Container for velocity calculations."""
    linear: float
    angular: float

@dataclass
class BinnedResult:
    """Container for binned analysis results."""
    file_name: str
    bin_label: str
    heading_angle: float
    velocities: VelocityResult

class BinnedHeadingAnalyzer:
    """
    Analyzes fly heading data with binning capabilities.
    
    Processes trajectory data to compute and bin heading changes,
    along with linear and angular velocities.
    """
    
    def __init__(self,
                 calibration_file: str,
                 bin_config: Optional[BinConfig] = None,
                 time_window: Optional[TimeWindow] = None):
        """
        Initialize the analyzer.
        
        Args:
            calibration_file: Path to calibration data
            bin_config: Configuration for binning
            time_window: Time window for analysis
        """
        self.calibration_df = pd.read_csv(calibration_file)
        self.bin_config = bin_config or BinConfig()
        self.time_window = time_window or TimeWindow()
        
    def calculate_velocities(self,
                           x_values: np.ndarray,
                           y_values: np.ndarray,
                           window: tuple[int, int]) -> Optional[VelocityResult]:
        """Calculate linear and angular velocities."""
        try:
            # Get window slices
            x_diff = np.diff(x_values)[window[0]:window[1]]
            y_diff = np.diff(y_values)[window[0]:window[1]]
            
            # Calculate heading for angular velocity
            heading = np.arctan2(y_diff, x_diff)
            
            # Calculate velocities
            linear_velocity = np.mean(np.sqrt(x_diff**2 + y_diff**2))
            angular_velocity = np.mean(np.diff(heading))
            
            # Normalize angular velocity
            angular_velocity = (angular_velocity + np.pi) % (2 * np.pi) - np.pi
            
            return VelocityResult(linear_velocity, angular_velocity)
            
        except Exception as e:
            print(f"Error calculating velocities: {e}")
            return None
            
    def assign_to_bin(self, angle_deg: float) -> Optional[str]:
        """Assign angle to appropriate bin."""
        edges = self.bin_config.bin_edges
        
        if self.bin_config.use_absolute_values:
            angle_deg = abs(angle_deg)
            while angle_deg > 180:
                angle_deg = 360 - angle_deg
                
        for i in range(len(edges) - 1):
            if edges[i] <= angle_deg < edges[i + 1]:
                return f"{edges[i]}-{edges[i+1]}°"
                
        return None
        
    def process_file(self, file_path: str) -> Optional[BinnedResult]:
        """Process a single trajectory file."""
        try:
            df = pd.read_csv(file_path)
            x_values = df['x'].values
            y_values = df['y'].values
            
            # Calculate velocities
            velocities = self.calculate_velocities(
                x_values,
                y_values,
                self.time_window.before_stimulus  # or after_stimulus
            )
            
            if not velocities:
                return None
                
            # Calculate heading difference (similar to heading_analyzer)
            # ... (reuse code from heading_analyzer)
            
            heading_deg = heading_rad * 57.29  # Convert to degrees
            
            # Assign to bin
            bin_label = self.assign_to_bin(heading_deg)
            if not bin_label:
                return None
                
            return BinnedResult(
                file_name=Path(file_path).name,
                bin_label=bin_label,
                heading_angle=heading_deg,
                velocities=velocities
            )
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
            
    def analyze_folder(self,
                      input_folder: str,
                      output_folder: str,
                      analysis_label: str) -> pd.DataFrame:
        """
        Analyze all files in a folder.
        
        Args:
            input_folder: Path to trajectory files
            output_folder: Path to save results
            analysis_label: Label for output files (e.g., 'before' or 'after')
        """
        results = []
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        for file_path in input_path.glob("*.csv"):
            result = self.process_file(str(file_path))
            if result:
                results.append({
                    "Bin": result.bin_label,
                    "diff_heading_Angle_degree": result.heading_angle,
                    "File Name": result.file_name,
                    "Angular_Velocity": result.velocities.angular,
                    "Linear_Velocity": result.velocities.linear
                })
                
        # Create DataFrame
        df = pd.DataFrame(results)
        if not df.empty:
            os.makedirs(output_path, exist_ok=True)
            df.to_csv(
                output_path / f"binned_analysis_{analysis_label}.csv",
                index=False
            )
            
        return df

def main():
    """Command line interface for binned analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze and bin fly heading data"
    )
    # ... (add command line arguments)
    
    args = parser.parse_args()
    # ... (implement command line interface)


if __name__ == "__main__":
    main()
