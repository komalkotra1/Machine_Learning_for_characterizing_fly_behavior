# src/stimulus_analysis/bin_preparation.py

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MergeConfig:
    """Configuration for merging binned and unbinned data."""
    key_column: str = 'File Name'
    merge_columns: List[str] = None
    
    def __post_init__(self):
        if self.merge_columns is None:
            self.merge_columns = [
                'difference_in_heading',
                'interp_heading',
                'heading'
            ]


class BinDataPreparator:
    """
    Prepares and merges binned and unbinned trajectory analysis data.
    
    Handles the combination of raw heading analysis results with
    binned categorization data.
    """
    
    def __init__(self, config: Optional[MergeConfig] = None):
        """
        Initialize the preparator.
        
        Args:
            config: Configuration for data merging
        """
        self.config = config or MergeConfig()
        
    def load_data(self, 
                 unbinned_path: str, 
                 binned_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both data files.
        
        Args:
            unbinned_path: Path to unbinned analysis results
            binned_path: Path to binned analysis results
            
        Returns:
            Tuple of (unbinned_df, binned_df)
        """
        try:
            unbinned_df = pd.read_csv(unbinned_path)
            binned_df = pd.read_csv(binned_path)
            
            # Clean the key column
            if self.config.key_column in binned_df.columns:
                binned_df[self.config.key_column] = binned_df[self.config.key_column].str.strip()
                
            return unbinned_df, binned_df
            
        except Exception as e:
            raise ValueError(f"Error loading data files: {e}")
            
    def merge_data(self, 
                  unbinned_df: pd.DataFrame, 
                  binned_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge binned and unbinned data.
        
        Args:
            unbinned_df: DataFrame with unbinned analysis results
            binned_df: DataFrame with binned categorizations
            
        Returns:
            Merged DataFrame
        """
        try:
            # Prepare columns for merging
            merge_cols = [col for col in self.config.merge_columns 
                         if col in unbinned_df.columns]
            
            if not merge_cols:
                raise ValueError("No valid columns found for merging")
                
            # Add file_name to merge columns if it exists
            merge_cols.append('file_name')
            
            # Perform merge
            merged_df = binned_df.merge(
                unbinned_df[merge_cols],
                left_on=self.config.key_column,
                right_on='file_name',
                how='left'
            )
            
            # Clean up merged data
            if 'file_name' in merged_df.columns:
                merged_df = merged_df.drop(columns=['file_name'])
                
            # Rename columns if needed
            if 'Angle' in merged_df.columns:
                merged_df = merged_df.rename(columns={'Angle': 'diff_heading_degree'})
                
            return merged_df
            
        except Exception as e:
            raise ValueError(f"Error merging data: {e}")
            
    def prepare_data(self,
                    unbinned_path: str,
                    binned_path: str,
                    output_path: str) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        Args:
            unbinned_path: Path to unbinned analysis results
            binned_path: Path to binned analysis results
            output_path: Path to save merged results
            
        Returns:
            Prepared DataFrame
        """
        # Load data
        unbinned_df, binned_df = self.load_data(unbinned_path, binned_path)
        
        # Merge data
        merged_df = self.merge_data(unbinned_df, binned_df)
        
        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        
        return merged_df


def main():
    """Command line interface for bin data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare and merge binned analysis data"
    )
    parser.add_argument(
        "--unbinned",
        required=True,
        help="Path to unbinned analysis results"
    )
    parser.add_argument(
        "--binned",
        required=True,
        help="Path to binned analysis results"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save merged results"
    )
    parser.add_argument(
        "--key-column",
        default="File Name",
        help="Column name for merging"
    )
    
    args = parser.parse_args()
    
    config = MergeConfig(key_column=args.key_column)
    preparator = BinDataPreparator(config)
    
    try:
        merged_df = preparator.prepare_data(
            args.unbinned,
            args.binned,
            args.output
        )
        print(f"Successfully processed {len(merged_df)} records")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
