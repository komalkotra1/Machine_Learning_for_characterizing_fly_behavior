# src/data_processing/trajectory_filter.py

import os
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class FilterConfig:
    """Configuration for trajectory filtering."""
    start_frame: int = 495
    end_frame: int = 661
    required_columns: List[str] = None

    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ['x', 'y']


@dataclass
class FilterResults:
    """Container for filtering results."""
    processed_files: int = 0
    valid_files: int = 0
    invalid_files: List[Tuple[str, str]] = None
    error_files: List[Tuple[str, str]] = None

    def __post_init__(self):
        self.invalid_files = self.invalid_files or []
        self.error_files = self.error_files or []

    def print_summary(self):
        """Print filtering results summary."""
        print(f"\nFiltering Summary:")
        print(f"Total files processed: {self.processed_files}")
        print(f"Valid files: {self.valid_files}")
        print(f"Invalid files: {len(self.invalid_files)}")
        print(f"Errors encountered: {len(self.error_files)}")
        
        if self.invalid_files:
            print("\nInvalid files (missing data):")
            for name, reason in self.invalid_files:
                print(f"  {name}: {reason}")
                
        if self.error_files:
            print("\nFiles with errors:")
            for name, error in self.error_files:
                print(f"  {name}: {error}")


class TrajectoryFilter:
    """
    Filters trajectory data based on data completeness criteria.
    
    Ensures trajectory data has valid coordinates within specified frame range.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize the filter with configuration.
        
        Args:
            config: Filtering configuration parameters
        """
        self.config = config or FilterConfig()
        
    def check_trajectory_validity(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check if trajectory data meets validity criteria.
        
        Args:
            data: DataFrame containing trajectory data
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check frame range
        frame_range = data[
            (data['i'] >= self.config.start_frame) & 
            (data['i'] <= self.config.end_frame)
        ]
        
        if len(frame_range) != (self.config.end_frame - self.config.start_frame + 1):
            return False, f"Missing frames in range {self.config.start_frame}-{self.config.end_frame}"
            
        # Check for NaN values in required columns
        for col in self.config.required_columns:
            if frame_range[col].isna().any():
                return False, f"Missing {col} values in frame range"
                
        return True, ""
        
    def process_file(self, file_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Process a single trajectory file.
        
        Args:
            file_path: Path to trajectory file
            
        Returns:
            Tuple of (filtered_data, error_message)
        """
        try:
            data = pd.read_csv(file_path)
            is_valid, reason = self.check_trajectory_validity(data)
            
            if is_valid:
                return data, None
            return None, reason
            
        except Exception as e:
            return None, str(e)
            
    def filter_trajectories(self, 
                          input_folder: str, 
                          output_folder: str) -> FilterResults:
        """
        Filter trajectory files from input folder.
        
        Args:
            input_folder: Path to folder containing trajectory files
            output_folder: Path to save filtered files
            
        Returns:
            FilterResults containing processing statistics
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        os.makedirs(output_path, exist_ok=True)
        
        results = FilterResults()
        
        for file_path in input_path.glob("*.csv"):
            results.processed_files += 1
            
            data, error = self.process_file(file_path)
            
            if error:
                if "Missing" in error:
                    results.invalid_files.append((file_path.name, error))
                else:
                    results.error_files.append((file_path.name, error))
                continue
                
            # Save valid file
            output_file = output_path / file_path.name
            data.to_csv(output_file, index=False)
            results.valid_files += 1
            
        return results


def main():
    """Command line interface for trajectory filtering."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter trajectory files based on data completeness"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input folder containing trajectory files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for filtered files"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=495,
        help="Start frame for validity check"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=661,
        help="End frame for validity check"
    )
    
    args = parser.parse_args()
    
    config = FilterConfig(start_frame=args.start_frame, end_frame=args.end_frame)
    filter = TrajectoryFilter(config)
    results = filter.filter_trajectories(args.input, args.output)
    results.print_summary()


if __name__ == "__main__":
    main()
