# src/saccade_detection/detector.py

import os
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass


@dataclass
class SaccadeParameters:
    """Configuration parameters for saccade detection."""
    height: float = 50.0        # Threshold for peak detection
    distance: int = 5          # Minimum samples between peaks
    smoothing_window: int = 3  # Window size for position smoothing


class SaccadeDetector:
    """
    Detects saccadic movements in fly trajectories using angular velocity analysis.
    
    This class processes trajectory data to identify rapid changes in direction
    that characterize saccadic flight behavior in Drosophila.
    """
    
    def __init__(self, params: Optional[SaccadeParameters] = None):
        """
        Initialize the saccade detector.
        
        Args:
            params: Configuration parameters for saccade detection.
                   If None, default parameters will be used.
        """
        self.params = params or SaccadeParameters()
        
    def get_angular_velocity(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate angular velocity from x,y position data.
        
        Args:
            df: DataFrame with 'x' and 'y' columns representing positions
            
        Returns:
            Array of angular velocities in radians/frame
        """
        dx = df['x'].diff().fillna(0)
        dy = df['y'].diff().fillna(0)
        angles = np.arctan2(dy, dx)
        angular_velocity = np.diff(angles, prepend=angles[0])
        return angular_velocity * 100  # Scale to amplify changes
    
    def detect_saccades(self, df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
        """
        Detect saccades in trajectory data.
        
        Args:
            df: DataFrame with 'x' and 'y' columns
            
        Returns:
            Tuple of (saccade indices array, statistics dictionary)
        """
        # Smooth the position data
        smoothed_df = df.copy()
        smoothed_df['x'] = df['x'].rolling(
            window=self.params.smoothing_window, 
            center=True
        ).mean().fillna(df['x'])
        smoothed_df['y'] = df['y'].rolling(
            window=self.params.smoothing_window, 
            center=True
        ).mean().fillna(df['y'])
        
        # Calculate angular velocity
        angvel = self.get_angular_velocity(smoothed_df)
        
        # Detect peaks in both directions
        neg_sac_idx, _ = find_peaks(
            -angvel, 
            height=self.params.height, 
            distance=self.params.distance
        )
        pos_sac_idx, _ = find_peaks(
            angvel, 
            height=self.params.height, 
            distance=self.params.distance
        )
        
        # Combine and sort all saccade indices
        saccade_indices = np.sort(np.concatenate((neg_sac_idx, pos_sac_idx)))
        
        # Calculate statistics
        stats = {
            'total_saccades': len(saccade_indices),
            'angular_velocity_stats': {
                'min': angvel.min(),
                'max': angvel.max(),
                'mean': angvel.mean(),
                'std': angvel.std()
            },
            'positive_saccades': len(pos_sac_idx),
            'negative_saccades': len(neg_sac_idx)
        }
        
        return saccade_indices, stats

    def process_files(
        self, 
        input_folder: str, 
        output_folder: str
    ) -> List[str]:
        """
        Process all CSV files in a folder for saccade detection.
        
        Args:
            input_folder: Path to folder containing trajectory CSV files
            output_folder: Path to save processed files
            
        Returns:
            List of successfully processed file paths
        """
        os.makedirs(output_folder, exist_ok=True)
        processed_files = []
        
        for filename in os.listdir(input_folder):
            if not filename.endswith(".csv"):
                continue
                
            try:
                file_path = os.path.join(input_folder, filename)
                df = pd.read_csv(file_path)
                
                saccade_indices, stats = self.detect_saccades(df)
                
                if len(saccade_indices) > 0:
                    output_path = os.path.join(output_folder, filename)
                    df.to_csv(output_path, index=False)
                    processed_files.append(output_path)
                    print(f"Processed {filename}: {stats['total_saccades']} saccades detected")
                else:
                    print(f"No saccades detected in {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                
        return processed_files


def main():
    """Command line interface for saccade detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect saccades in fly trajectories")
    parser.add_argument("input_folder", help="Folder containing trajectory CSVs")
    parser.add_argument("output_folder", help="Folder to save processed files")
    parser.add_argument("--height", type=float, default=50.0, help="Peak detection threshold")
    parser.add_argument("--distance", type=int, default=5, help="Minimum samples between peaks")
    
    args = parser.parse_args()
    
    params = SaccadeParameters(height=args.height, distance=args.distance)
    detector = SaccadeDetector(params)
    detector.process_files(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
