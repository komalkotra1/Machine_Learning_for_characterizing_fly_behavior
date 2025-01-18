# Machine_Learning_for_characterizing_fly_behavior

Analysis pipeline for studying free flight behavior in Drosophila melanogaster using multi-camera tracking and behavioral analysis.

## Project Overview
This project processes and analyzes free flight behavior of Drosophila in a chamber with looming stimulus, using multi-camera tracking (Braid) for precise 3D position data.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Machine_Learning_for_characterizing_fly_behavior.git

# Install required packages
pip install -r requirements.txt
src/
├── video_analysis/       # Video processing and blob detection
└── saccade_detection/   # Saccade identification from trajectories
Modules
1. Video Analysis (src/video_analysis/)
Processes video data to extract fly positions using blob detection.
Key features:

Multi-video batch processing
Blob detection for tracking fly positions
CSV output of trajectory data
Debug mode for visualization

Usage:
pythonCopyfrom video_analysis import VideoProcessor

processor = VideoProcessor(debug=False, overwrite_csv=False)
processor.process_video("path/to/video.mp4")







Currently in progress.
This is the working repository for my master thesis project, where I am trying to classify how flies behave in a free-flying setup which is developed in the NFC group of MPINB, Bonn. The project is in progress at the moment. The goal is to characterize different flight behaviors in the presence of a looming stimulus using machine learning algorithm. 
For any questions, shoot me an email at any of the email-adresses below:

komalkotra1@gmail.com

s6kokoma@uni-bonn.de

komal.kotra@mpinb.mpg.de
