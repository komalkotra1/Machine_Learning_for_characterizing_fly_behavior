# Machine Learning for Characterizing Fly Behavior

Analysis pipeline for studying free flight behavior in Drosophila melanogaster using multi-camera tracking and behavioral analysis.

## Project Overview

This project processes and analyzes free flight behavior of Drosophila in a chamber with looming stimulus, using multi-camera tracking (Braid) for precise 3D position data. This is my masters thesis project and is currently ongoing, so keep your eyes for the recent commits.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Machine_Learning_for_characterizing_fly_behavior.git

# Install required packages
pip install -r requirements.txt
```
### Usage

1. Prepare your data:
   - Place your video files in a designated directory
   - Ensure you have the calibration file ready
   - Create an output directory for results

2. Run the complete analysis pipeline:
   The main.py script provides a complete analysis pipeline that runs all componenets in sequence.
```bash
python main.py --video-dir /path/to/videos --output-dir /path/to/output --calibration /path/to/calibration.csv
```
Available Arguments:
```bash
--video-dir         Directory containing input videos
--output-dir        Directory for analysis outputs
--calibration       Path to calibration file
--saccade-threshold Threshold for saccade detection (default: 50.0)
```


Alternative: Run Specific components:

Process videos only:
```bash
python -m src.video_analysis.video_processor --input /path/to/videos --output /path/to/tracking
```
Detect saccades:
```bash
python -m src.saccade_detection.detector --input /path/to/tracking --output /path/to/saccades
```
Full heading analysis:
```bash
python -m src.stimulus_analysis.heading_analyzer --input /path/to/data --output /path/to/results
```



## Project Structure

```
machine_learning_for_characterizing_fly_behavior/
├── main.py                  # Main analysis pipeline
├── requirements.txt         # Project dependencies
├── README.md               # Project documentation
└── src/                    # Source code modules
    ├── video_analysis/           # Video processing and blob detection
       ├── init.py
       └── video_processor.py    # Process video files for tracking
    ├── saccade_detection/       # Saccade identification from trajectories
       ├── init.py
       └── detector.py          # Detect saccadic movements
    ├── data_processing/         # Data organization and processing
       ├── init.py
       ├── trajectory_splitter.py    # Separate saccade/non-saccade data
       ├── tracking_file_processor.py # Process tracking files
       ├── trajectory_filter.py       # Filter valid trajectories
       ├── file_matcher.py           # Match and merge related files
       └── bin_segmenter.py          # Segment data into bin ranges
   ├──stimulus_analysis/       # Stimulus response analysis
      ├── init.py
      ├── heading_analyzer.py   # Basic heading analysis
      ├── binned_analyzer.py    # Binned heading analysis
      ├── bin_preparation.py    # Prepare and merge binned data
      ├── visualization.py      # Basic visualization tools
      └── bin_visualizer.py     # Comparative visualization tools

```

## Modules

### 1. Video Analysis (`src/video_analysis/`)

Processes video data to extract fly positions using blob detection.

#### Key features:
* Multi-video batch processing
* Blob detection for tracking fly positions
* CSV output of trajectory data
* Debug mode for visualization

#### Usage:
```python
from video_analysis import VideoProcessor

processor = VideoProcessor(debug=False, overwrite_csv=False)
processor.process_video("path/to/video.mp4")
```

### 2. Saccade Detection (`src/saccade_detection/`)

Analyzes trajectory data to identify saccadic flight maneuvers.

#### Key features:
* Angular velocity calculation
* Peak detection for saccade identification
* Configurable detection parameters
* Batch processing of trajectory files

#### Usage:
```python
from saccade_detection.detector import SaccadeDetector, SaccadeParameters

# Configure detection parameters
params = SaccadeParameters(
    height=50.0,          # Saccade detection threshold
    distance=5,           # Minimum frames between saccades
    smoothing_window=3    # Position smoothing window
)

# Create detector and process files
detector = SaccadeDetector(params)
detector.process_files("input/folder", "output/folder")
```
### 3. Trajectory Data Processing (`src/data_processing/`)

Manages the organization and processing of trajectory data files.

#### Key features:
* Separates tracking files into saccade and non-saccade sets
* Handles file organization and copying
* Error handling and reporting
* Command-line interface for batch processing

#### Usage:
```python
from data_processing.trajectory_splitter import TrajectoryDataSplitter

splitter = TrajectoryDataSplitter()
copied, failed = splitter.process(
    tracking_folder="path/to/tracking",
    saccade_folder="path/to/saccade_detected",
    non_saccade_folder="path/to/non_saccade"
)
print(f"Processed {copied} files")
```
#### Tracking File Processor
A component of the data processing module that handles file organization.

##### Features:
* Compares contents between tracking and saccade folders
* Identifies and extracts non-saccade files
* Provides detailed processing statistics
* Flexible file pattern matching

##### Usage:
```python
from data_processing.tracking_file_processor import TrackingFileProcessor

processor = TrackingFileProcessor()
results = processor.process(
    saccade_folder="path/to/saccade",
    tracking_folder="path/to/tracking",
    output_folder="path/to/output"
)
results.print_summary()
```
#### Trajectory Filter
Quality control component that filters trajectory data based on completeness criteria.

##### Features:
* Validates data completeness within specified frame ranges
* Checks for missing coordinate values
* Detailed filtering statistics
* Configurable validation parameters

##### Usage:
```python
from data_processing.trajectory_filter import TrajectoryFilter, FilterConfig

config = FilterConfig(
    start_frame=495,
    end_frame=661,
    required_columns=['x', 'y']
)

filter = TrajectoryFilter(config)
results = filter.filter_trajectories(
    input_folder="path/to/input",
    output_folder="path/to/output"
)
results.print_summary()
```

#### Heading Analyzer
Analyzes fly heading changes in response to visual stimuli.

##### Features:
* Computes heading before and after stimulus
* Calibration-based heading interpolation
* Configurable analysis time windows
* Automated batch processing
* Results visualization

##### Usage:
```python
from stimulus_analysis.heading_analyzer import HeadingAnalyzer, TimeWindow

# Configure analysis windows
time_window = TimeWindow(
    before_stimulus=(495, 535),
    after_stimulus=(620, 660),
    fps=25
)

# Create analyzer
analyzer = HeadingAnalyzer(
    calibration_file="path/to/calibration.csv",
    time_window=time_window
)

# Process files
results_df = analyzer.analyze_folder(
    input_folder="path/to/trajectories",
    output_folder="path/to/results"
)
```
#### Binned Analysis
Advanced analysis component for categorizing heading changes.

##### Features:
* Flexible binning configurations
* Absolute or relative angle binning
* Integrated velocity calculations
* Support for before/after stimulus analysis
* Detailed result categorization

##### Usage:
```python
from stimulus_analysis.binned_analyzer import BinnedHeadingAnalyzer, BinConfig, TimeWindow

# Configure analysis
bin_config = BinConfig(
    bin_edges=[0, 45, 90, 135, 180],
    use_absolute_values=True
)

time_window = TimeWindow(
    before_stimulus=(495, 535),
    after_stimulus=(620, 660)
)

# Create analyzer
analyzer = BinnedHeadingAnalyzer(
    calibration_file="path/to/calibration.csv",
    bin_config=bin_config,
    time_window=time_window
)

# Analyze before stimulus
before_results = analyzer.analyze_folder(
    input_folder="path/to/input",
    output_folder="path/to/output",
    analysis_label="before"
)

# Update time window and analyze after stimulus
analyzer.time_window.active_window = "after"
after_results = analyzer.analyze_folder(
    input_folder="path/to/input",
    output_folder="path/to/output",
    analysis_label="after"
)
```

#### Bin Data Preparation
Data preparation utility for merging binned and unbinned analysis results.

##### Features:
* Merges raw heading data with binned categorizations
* Flexible column mapping
* Data validation and cleaning
* Configurable merge operations

##### Usage:
```python
from stimulus_analysis.bin_preparation import BinDataPreparator, MergeConfig

# Configure merge operation
config = MergeConfig(
    key_column='File Name',
    merge_columns=['difference_in_heading', 'interp_heading', 'heading']
)

# Create preparator and process data
preparator = BinDataPreparator(config)
merged_df = preparator.prepare_data(
    unbinned_path="path/to/unbinned.csv",
    binned_path="path/to/binned.csv",
    output_path="path/to/output.csv"
)
```
#### File Matcher
Utility for matching and merging related data files from different analysis stages.

##### Features:
* Matches files based on common identifiers
* Provides detailed matching statistics
* Flexible column matching
* Safe data loading and validation

##### Usage:
```python
from data_processing.file_matcher import FileMatcher

matcher = FileMatcher(match_column='File Name')
results = matcher.match_files(
    first_file="path/to/first.csv",
    second_file="path/to/second.csv",
    output_file="path/to/output.csv"
)
```
# View matching statistics
results.print_summary()

#### Bin Matcher
Specialized utility for processing and matching data based on bin ranges.

##### Features:
* Configurable bin range definitions
* Flexible bin parsing and matching
* Automated file organization by bin
* Data matching across analysis stages
* Detailed processing reports

##### Usage:
```python
from data_processing.bin_matcher import BinMatcher, BinRange

# Create matcher with custom bin ranges
matcher = BinMatcher([
    BinRange(0, 45, '0_45'),
    BinRange(45, 90, '90_45'),
    BinRange(90, 135, '135_90'),
    BinRange(135, 180, '180_135')
])

# Process bins
output_files = matcher.process_bins(
    input_file="path/to/input.csv",
    output_dir="path/to/output"
)

# Match with reference data
matched_df = matcher.match_bin_data(
    bin_file="path/to/bin_file.csv",
    reference_file="path/to/reference.csv",
    output_file="path/to/matched.csv"
)
```
#### Bin Segmenter
Utility for segmenting data into predefined bin ranges.

##### Features:
* Configurable bin segment definitions
* Automated file organization by segment
* Flexible bin range parsing
* Detailed segmentation reporting

##### Usage:
```python
from data_processing.bin_segmenter import BinSegmenter, BinSegment

# Create segmenter with custom ranges
segmenter = BinSegmenter([
    BinSegment(0, 45, '0_45'),
    BinSegment(45, 90, '45_90'),
    BinSegment(90, 135, '90_135'),
    BinSegment(135, 180, '135_180')
])

# Segment data
segment_files = segmenter.segment_data(
    input_file="path/to/input.csv",
    output_dir="path/to/output"
)
```

#### Bin Comparison Visualizer
Advanced visualization tool for comparing before/after stimulus responses.

##### Features:
* Multiple plot types (line, scatter, bar)
* Customizable plot styling
* Automatic plot layout management
* High-resolution output
* Comprehensive data comparison

##### Usage:
```python
from stimulus_analysis.bin_visualizer import BinComparisonVisualizer, PlotStyles

# Configure visualization style
styles = PlotStyles(
    figsize=(18, 6),
    before_color='orange',
    after_color='red',
    scatter_color='purple',
    dpi=300
)

# Create visualizer and generate plots
visualizer = BinComparisonVisualizer(styles)
visualizer.visualize_comparison(
    input_file="path/to/comparison.csv",
    output_file="path/to/output.png"
)
```
## Dependencies

* numpy>=1.21.0: Array operations and numerical computations
* pandas>=1.3.0: Data handling and CSV processing
* scipy>=1.7.0: Signal processing for saccade detection
* matplotlib>=3.4.0: Visualization and plotting
* seaborn>=0.11.0: Enhanced visualization
* tqdm>=4.62.0: Progress bars for batch processing
* pathlib: Path handling and file operations
* dataclasses: Data structure organization
* re: Regular expression operations
## Contact
Currently in progress.
This is the working repository for my master thesis project, where I am trying to classify how flies behave in a free-flying setup which is developed in the NFC group of MPINB, Bonn. The project is in progress at the moment. The goal is to characterize different flight behaviors in the presence of a looming stimulus using machine learning algorithm. 
For any questions, shoot me an email at any of the email-adresses below:

komalkotra1@gmail.com

s6kokoma@uni-bonn.de

komal.kotra@mpinb.mpg.de
