# src/data_processing/bin_segmenter.py

import os
import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class BinSegment:
    """Configuration for a bin segment."""
    min_value: int
    max_value: int
    label: str

    def __str__(self) -> str:
        return f"{self.min_value}_{self.max_value}"


class BinSegmenter:
    """
    Segments data into predefined bin ranges.
    
    Processes trajectory data and segments it into separate files
    based on specified bin ranges.
    """
    
    def __init__(self, segments: Optional[List[BinSegment]] = None):
        """
        Initialize the segmenter.
        
        Args:
            segments: List of bin segment configurations
        """
        self.segments = segments or [
            BinSegment(0, 45, '0_45'),
            BinSegment(45, 90, '45_90'),
            BinSegment(90, 135, '90_135'),
            BinSegment(135, 180, '135_180')
        ]
        
    @staticmethod
    def parse_bin_range(bin_str: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract numeric ranges from bin string."""
        match = re.search(r'(\d+)-(\d+)', bin_str)
        if match:
            start, end = map(int, match.groups())
            return min(start, end), max(start, end)
        return None, None
        
    def filter_segment(self, 
                      data: pd.DataFrame, 
                      segment: BinSegment) -> pd.DataFrame:
        """
        Filter data for specific bin segment.
        
        Args:
            data: DataFrame to filter
            segment: Target bin segment
            
        Returns:
            Filtered DataFrame
        """
        return data[data['Bin'].apply(
            lambda x: self._check_segment_match(x, segment)
        )]
        
    def _check_segment_match(self, 
                           bin_str: str, 
                           segment: BinSegment) -> bool:
        """Check if bin string matches target segment."""
        start, end = self.parse_bin_range(bin_str)
        if start is None or end is None:
            return False
        return start >= segment.min_value and end <= segment.max_value
        
    def segment_data(self,
                    input_file: str,
                    output_dir: str) -> Dict[str, Path]:
        """
        Segment data into bin ranges.
        
        Args:
            input_file: Path to input CSV
            output_dir: Directory for segmented files
            
        Returns:
            Dictionary mapping segment labels to output files
        """
        try:
            # Load data
            df = pd.read_csv(input_file)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            segment_files = {}
            
            # Process each segment
            for segment in self.segments:
                # Filter data
                filtered_df = self.filter_segment(df, segment)
                
                # Save segment
                output_file = output_path / f"outputfile_bins_{segment}.csv"
                filtered_df.to_csv(output_file, index=False)
                
                segment_files[str(segment)] = output_file
                
            return segment_files
            
        except Exception as e:
            raise ValueError(f"Error segmenting data: {e}")


def main():
    """Command line interface for bin segmentation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Segment trajectory data by bin ranges"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file to segment"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for segmented files"
    )
    
    args = parser.parse_args()
    
    segmenter = BinSegmenter()
    
    try:
        segment_files = segmenter.segment_data(
            args.input,
            args.output_dir
        )
        
        print("\nSegmentation complete.")
        print("\nSegmented files:")
        for segment, file_path in segment_files.items():
            print(f"  {segment}: {file_path}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
