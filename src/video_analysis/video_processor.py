# src/video_analysis/video_processor.py
import argparse
import os
import sys
from glob import glob
from typing import List, Optional
from tqdm import tqdm

class VideoProcessor:
    """
    Process videos for fly tracking using blob detection.
    
    This class handles both single video files and folders of videos,
    converting them to CSV tracking data.
    """
    
    def __init__(self, debug: bool = False, overwrite_csv: bool = False):
        """
        Initialize the video processor.
        
        Args:
            debug (bool): Enable debug mode for additional output
            overwrite_csv (bool): Whether to overwrite existing CSV files
        """
        self.debug = debug
        self.overwrite_csv = overwrite_csv
        
    def process_video(self, video_path: str) -> Optional[str]:
        """
        Process a single video file and convert to CSV.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Path to the output CSV file if successful, None otherwise
        """
        try:
            from analysis.video_tools import process_video
            return process_video(
                video_path, 
                debug=self.debug, 
                overwrite_csv=self.overwrite_csv
            )
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return None
            
    def process_folder(self, folder_path: str) -> List[str]:
        """
        Process all videos in a folder.
        
        Args:
            folder_path (str): Path to folder containing videos
            
        Returns:
            List[str]: List of successfully processed CSV file paths
        """
        print(f"Processing folder: {folder_path}")
        video_files = glob(os.path.join(folder_path, "*.mp4"))
        processed_files = []
        
        for file in tqdm(video_files):
            result = self.process_video(file)
            if result:
                processed_files.append(result)
                
        return processed_files

def main():
    """Command line interface for video processing."""
    parser = argparse.ArgumentParser(
        description="Process videos for fly tracking analysis"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input video file or folder"
    )
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files",
        default=False
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode",
        default=False
    )
    
    args = parser.parse_args()
    processor = VideoProcessor(debug=args.debug, overwrite_csv=args.overwrite)
    
    try:
        if os.path.isfile(args.input):
            result = processor.process_video(args.input)
            if result:
                print(f"Successfully processed: {args.input}")
        elif os.path.isdir(args.input):
            results = processor.process_folder(args.input)
            print(f"Successfully processed {len(results)} videos")
        else:
            raise ValueError("Input path is not a valid file or directory")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
