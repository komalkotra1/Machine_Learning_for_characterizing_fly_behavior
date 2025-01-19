# src/data_processing/bin_matcher.py

import os
import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class BinRange:
    """Configuration for bin ranges."""
    min_value: int
    max_value: int
    label: str

    def contains(self, start: int, end: int) -> bool:
        """Check if a range falls within this bin."""
        return self.min_value <= start and end <= self.max_value


class BinMatcher:
    """
    Matches and filters data based on bin ranges.
    
    Processes trajectory data files based on specified bin ranges
    and matches corresponding data across analysis stages.
    """
    
    def __init__(self, bin_ranges: Optional[List[BinRange]] = None):
        """
        Initialize the matcher with bin configurations.
        
        Args:
            bin_ranges: List of bin range configurations
        """
        self.bin_ranges = bin_ranges or [
            BinRange(0, 45, '0_45'),
            BinRange(45, 90, '90_45'),
            BinRange(90, 135, '135_90'),
            BinRange(135, 180, '180_135')
        ]
        
    @staticmethod
    def parse_bin_range(bin_str: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract numeric ranges from bin string."""
        match = re.search(r'(\d+)-(\d+)', bin_str)
        if match:
            start, end = map(int, match.groups())
            return min(start, end), max(start, end)
        return None, None
        
    def filter_by_bin(self, 
                     data: pd.DataFrame, 
                     bin_range: BinRange) -> pd.DataFrame:
        """
        Filter data for specific bin range.
        
        Args:
            data: DataFrame to filter
            bin_range: Target bin configuration
            
        Returns:
            Filtered DataFrame
        """
        return data[data['Bin'].apply(
            lambda x: self._check_bin_match(x, bin_range)
        )]
        
    def _check_bin_match(self, 
                        bin_str: str, 
                        bin_range: BinRange) -> bool:
        """Check if bin string matches target range."""
        start, end = self.parse_bin_range(bin_str)
        if start is None or end is None:
            return False
        return bin_range.contains(start, end)
        
    def process_bins(self,
                    input_file: str,
                    output_dir: str) -> Dict[str, str]:
        """
        Process data for all bin ranges.
        
        Args:
            input_file: Path to input CSV
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping bin labels to output files
        """
        try:
            df = pd.read_csv(input_file)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            output_files = {}
            
            for bin_range in self.bin_ranges:
                filtered_df = self.filter_by_bin(df, bin_range)
                output_file = output_path / f"outputfile_bins_{bin_range.label}.csv"
                filtered_df.to_csv(output_file, index=False)
                output_files[bin_range.label] = str(output_file)
                
            return output_files
            
        except Exception as e:
            raise ValueError(f"Error processing bins: {e}")
            
    def match_bin_data(self,
                      bin_file: str,
                      reference_file: str,
                      output_file: str,
                      match_column: str = 'File Name') -> pd.DataFrame:
        """
        Match binned data with reference data.
        
        Args:
            bin_file: Path to binned data
            reference_file: Path to reference data
            output_file: Path for matched output
            match_column: Column to use for matching
            
        Returns:
            Matched DataFrame
        """
        try:
            # Load data
            bin_df = pd.read_csv(bin_file)
            ref_df = pd.read_csv(reference_file)
            
            # Perform match
            matched_df = bin_df.merge(ref_df, on=match_column)
            
            # Save results
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            matched_df.to_csv(output_file, index=False)
            
            return matched_df
            
        except Exception as e:
            raise ValueError(f"Error matching data: {e}")


def main():
    """Command line interface for bin matching."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process and match binned trajectory data"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for binned files"
    )
    parser.add_argument(
        "--reference",
        help="Reference file for matching"
    )
    parser.add_argument(
        "--match-output",
        help="Output file for matched data"
    )
    
    args = parser.parse_args()
    
    matcher = BinMatcher()
    
    try:
        # Process bins
        output_files = matcher.process_bins(
            args.input,
            args.output_dir
        )
        print("Bin processing complete:")
        for label, path in output_files.items():
            print(f"  {label}: {path}")
            
        # Match data if requested
        if args.reference and args.match_output:
            for bin_file in output_files.values():
                matched_df = matcher.match_bin_data(
                    bin_file,
                    args.reference,
                    args.match_output
                )
                print(f"\nMatched {len(matched_df)} rows")
                print(f"Saved to: {args.match_output}")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
