# src/data_processing/tracking_file_processor.py

import os
import shutil
from pathlib import Path
from typing import Set, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ProcessingResult:
    """Container for file processing results."""
    copied_files: List[str]
    failed_files: List[tuple[str, str]]
    total_processed: int

    def print_summary(self) -> None:
        """Print a summary of the processing results."""
        print(f"\nProcessing Summary:")
        print(f"Total files processed: {self.total_processed}")
        print(f"Successfully copied: {len(self.copied_files)}")
        if self.failed_files:
            print("\nFailed copies:")
            for file_name, error in self.failed_files:
                print(f"  {file_name}: {error}")


class TrackingFileProcessor:
    """
    Processes tracking files by comparing contents of two folders and
    extracting unique files for further analysis.
    """
    
    def __init__(self, file_pattern: str = "*.csv"):
        """
        Initialize the processor.
        
        Args:
            file_pattern: Pattern to match files (default: "*.csv")
        """
        self.file_pattern = file_pattern
        
    def setup_folders(self, 
                     saccade_folder: str, 
                     tracking_folder: str,
                     output_folder: str) -> None:
        """
        Set up and validate folder paths.
        
        Args:
            saccade_folder: Path to folder with saccade files
            tracking_folder: Path to folder with all tracking files
            output_folder: Path to save non-saccade files
        """
        self.saccade_folder = Path(saccade_folder)
        self.tracking_folder = Path(tracking_folder)
        self.output_folder = Path(output_folder)
        
        if not self.saccade_folder.exists():
            raise ValueError(f"Saccade folder does not exist: {saccade_folder}")
        if not self.tracking_folder.exists():
            raise ValueError(f"Tracking folder does not exist: {tracking_folder}")
            
        # Create output folder if needed
        os.makedirs(self.output_folder, exist_ok=True)
        
    def get_file_sets(self) -> tuple[Set[str], Set[str]]:
        """Get sets of files from both folders."""
        saccade_files = {
            f.name for f in self.saccade_folder.glob(self.file_pattern)
        }
        tracking_files = {
            f.name for f in self.tracking_folder.glob(self.file_pattern)
        }
        return saccade_files, tracking_files
        
    def copy_files(self, files_to_copy: Set[str]) -> ProcessingResult:
        """
        Copy files to output folder.
        
        Args:
            files_to_copy: Set of filenames to copy
            
        Returns:
            ProcessingResult with copy statistics
        """
        copied_files = []
        failed_files = []
        
        for filename in files_to_copy:
            try:
                source = self.tracking_folder / filename
                destination = self.output_folder / filename
                shutil.copy(source, destination)
                copied_files.append(filename)
            except Exception as e:
                failed_files.append((filename, str(e)))
                
        return ProcessingResult(
            copied_files=copied_files,
            failed_files=failed_files,
            total_processed=len(files_to_copy)
        )
        
    def process(self,
               saccade_folder: str,
               tracking_folder: str,
               output_folder: str) -> ProcessingResult:
        """
        Process tracking files and extract non-saccade files.
        
        Args:
            saccade_folder: Path to folder with saccade files
            tracking_folder: Path to folder with all tracking files
            output_folder: Path to save non-saccade files
            
        Returns:
            ProcessingResult containing processing statistics
        """
        self.setup_folders(saccade_folder, tracking_folder, output_folder)
        
        # Get file sets and find unique files
        saccade_files, tracking_files = self.get_file_sets()
        files_to_copy = tracking_files - saccade_files
        
        # Copy files and return results
        return self.copy_files(files_to_copy)


def main():
    """Command-line interface for the tracking file processor."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process tracking files and extract non-saccade data"
    )
    parser.add_argument(
        "--saccade",
        required=True,
        help="Path to folder containing saccade files"
    )
    parser.add_argument(
        "--tracking",
        required=True,
        help="Path to folder containing all tracking files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save non-saccade files"
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File pattern to match (default: *.csv)"
    )
    
    args = parser.parse_args()
    
    processor = TrackingFileProcessor(file_pattern=args.pattern)
    results = processor.process(args.saccade, args.tracking, args.output)
    results.print_summary()


if __name__ == "__main__":
    main()
