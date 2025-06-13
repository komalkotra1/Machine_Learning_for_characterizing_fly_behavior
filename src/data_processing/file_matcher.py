# src/data_processing/file_matcher.py

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class MatchResult:
    """Container for file matching results."""
    total_first: int
    total_second: int
    common_files: int
    matched_data: pd.DataFrame

    def print_summary(self):
        """Print matching results summary."""
        print(f"\nMatching Summary:")
        print(f"Files in first dataset: {self.total_first}")
        print(f"Files in second dataset: {self.total_second}")
        print(f"Common files matched: {self.common_files}")


class FileMatcher:
    """
    Matches and merges related CSV files based on common identifiers.
    
    Handles the matching of files from different analysis stages
    while maintaining data integrity.
    """
    
    def __init__(self, match_column: str = 'File Name'):
        """
        Initialize the matcher.
        
        Args:
            match_column: Column name to use for matching
        """
        self.match_column = match_column
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV data safely.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            if self.match_column not in df.columns:
                raise ValueError(f"Match column '{self.match_column}' not found")
            return df
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")
            
    def match_files(self,
                   first_file: str,
                   second_file: str,
                   output_file: Optional[str] = None) -> MatchResult:
        """
        Match and merge files based on common identifiers.
        
        Args:
            first_file: Path to first CSV file
            second_file: Path to second CSV file
            output_file: Optional path to save merged results
            
        Returns:
            MatchResult containing matching statistics
        """
        # Load data
        df1 = self.load_data(first_file)
        df2 = self.load_data(second_file)
        
        # Perform merge
        merged_df = pd.merge(
            df1,
            df2,
            on=self.match_column,
            how='inner'
        )
        
        # Save if output path provided
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_csv(output_file, index=False)
            
        # Return results
        return MatchResult(
            total_first=len(df1),
            total_second=len(df2),
            common_files=len(merged_df),
            matched_data=merged_df
        )


def main():
    """Command line interface for file matching."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Match and merge CSV files based on common identifiers"
    )
    parser.add_argument(
        "--first",
        required=True,
        help="Path to first CSV file"
    )
    parser.add_argument(
        "--second",
        required=True,
        help="Path to second CSV file"
    )
    parser.add_argument(
        "--output",
        help="Path to save merged results"
    )
    parser.add_argument(
        "--match-column",
        default="File Name",
        help="Column name to use for matching"
    )
    
    args = parser.parse_args()
    
    matcher = FileMatcher(match_column=args.match_column)
    
    try:
        results = matcher.match_files(
            args.first,
            args.second,
            args.output
        )
        results.print_summary()
        
        if args.output:
            print(f"\nMatched data saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
