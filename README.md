# Machine Learning for Characterizing Fly Behavior

Analysis pipeline for studying free flight behavior in Drosophila melanogaster using multi-camera tracking and behavioral analysis.

## Project Overview

This project processes and analyzes free flight behavior of Drosophila in a chamber with looming stimulus, using multi-camera tracking (Braid) for precise 3D position data.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Machine_Learning_for_characterizing_fly_behavior.git

# Install required packages
pip install -r requirements.txt
```

## Project Structure

```
src/
├── video_analysis/           # Video processing and blob detection
│   ├── init.py
│   └── video_processor.py    # Process video files for tracking
├── saccade_detection/       # Saccade identification from trajectories
│   ├── init.py
│   └── detector.py          # Detect saccadic movements
├── data_processing/         # Data organization and processing
│   ├── init.py
│   ├── trajectory_splitter.py    # Separate saccade/non-saccade data
│   ├── tracking_file_processor.py # Process tracking files
│   ├── trajectory_filter.py       # Filter valid trajectories
│   └── file_matcher.py           # Match and merge related files
└── stimulus_analysis/       # Stimulus response analysis
      ├── init.py
      ├── heading_analyzer.py   # Basic heading analysis
      ├── binned_analyzer.py    # Binned heading analysis
      ├── bin_preparation.py    # Prepare and merge binned data
      └── visualization.py      # Data visualization tools
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

## Dependencies

* numpy: Array operations and numerical computations
* pandas: Data handling and CSV processing
* scipy: Signal processing for saccade detection
* tqdm: Progress bars for batch processing
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
