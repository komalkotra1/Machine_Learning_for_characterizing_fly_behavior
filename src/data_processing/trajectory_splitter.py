# src/data_processing/trajectory_splitter.py

import os
import shutil
from typing import List, Set, Optional
from pathlib import Path


class TrajectoryDataSplitter:
    """
    Splits trajectory data into saccade and non-saccade files based on detection results.
    
    This class manages the separation of tracking files into two categories:
    - Files containing detected saccades
    - Files without detected saccades
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the splitter with an optional base path.
        
        Args:
            base_path: Base directory containing all data folders
        """
        self.base_path = Path(base_path) if base_path else None
        
    def setup_folders(self, 
                     tracking_folder: str, 
                     saccade_folder: str, 
                     non_saccade_folder: str) -> None:
        """
        Set up the necessary folder structure.
        
        Args:
            tracking_folder: Path to original tracking files
            saccade_folder: Path to files with detected saccades
            non_saccade_folder: Path to save files without saccades
        """
        self.tracking_folder = Path(tracking_folder)
        self.saccade_folder = Path(saccade_folder)
        self.non_saccade_folder = Path(non_saccade_folder)
        
        # Create output folder if it doesn't exist
        os.makedirs(self.non_saccade_folder, exist_ok=True)
        
    def get_file_sets(self) -> tuple[Set[str], Set[str]]:
        """
        Get sets of files from saccade and tracking folders.
        
        Returns:
            Tuple of (saccade files set, tracking files set)
        """
        saccade_files = set(os.listdir(self.saccade_folder))
        tracking_files = set(os.listdir(self.tracking_folder))
        return saccade_files, tracking_files
        
    def identify_non_saccade_files(self) -> List[str]:
        """
        Identify files that don't contain saccades.
        
        Returns:
            List of filenames without detected saccades
        """
        saccade_files, tracking_files = self.get_file_sets()
        return list(tracking_files - saccade_files)
        
    def copy_non_saccade_files(self) -> tuple[int, List[str]]:
        """
        Copy files without saccades to the non-saccade folder.
        
        Returns:
            Tuple of (number of copied files, list of any failed copies)
        """
        non_saccade_files = self.identify_non_saccade_files()
        failed_copies = []
        successful_copies = 0
        
        for file_name in non_saccade_files:
            try:
                source_path = self.tracking_folder / file_name
                destination_path = self.non_saccade_folder / file_name
                shutil.copy(source_path, destination_path)
                successful_copies += 1
            except Exception as e:
                failed_copies.append((file_name, str(e)))
                
        return successful_copies, failed_copies
        
    def process(self, 
                tracking_folder: str, 
                saccade_folder: str, 
                non_saccade_folder: str) -> tuple[int, List[str]]:
        """
        Complete processing pipeline for splitting trajectory data.
        
        Args:
            tracking_folder: Path to original tracking files
            saccade_folder: Path to files with detected saccades
            non_saccade_folder: Path to save files without saccades
            
        Returns:
            Tuple of (number of processed files, list of any errors)
        """
        self.setup_folders(tracking_folder, saccade_folder, non_saccade_folder)
        return self.copy_non_saccade_files()


def main():
    """Command line interface for trajectory splitting."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split trajectory files into saccade and non-saccade sets"
    )
    parser.add_argument(
        "--tracking", 
        required=True, 
        help="Path to original tracking files"
    )
    parser.add_argument(
        "--saccade", 
        required=True, 
        help="Path to files with detected saccades"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path for files without saccades"
    )
    
    args = parser.parse_args()
    
    splitter = TrajectoryDataSplitter()
    copied, failed = splitter.process(
        args.tracking, 
        args.saccade, 
        args.output
    )
    
    print(f"Successfully copied {copied} files")
    if failed:
        print("Failed copies:")
        for file_name, error in failed:
            print(f"  {file_name}: {error}")


if __name__ == "__main__":
    main()
