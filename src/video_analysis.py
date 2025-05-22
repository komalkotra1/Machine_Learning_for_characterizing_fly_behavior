#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import glob
import os
import re
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import circmean
from sklearn.preprocessing import StandardScaler
from tqdm.contrib.concurrent import process_map
from tqdm.notebook import tqdm
import pathlib
from extract_stimulus_heading_for_camera import (
    process_braidz_file,
    create_interpolation_function,
)
from natsort import natsorted
from datetime import datetime


# In[3]:


# Files for first strain Canton-s
canton_s = [
    "20250501_183152.braidz",
    "20250429_193010.braidz",
    "20250411_175919.braidz",
    "20250111_151946.braidz",
    "20241126_171308.braidz",
    "20250110_142709.braidz",
    "20241130_161634.braidz",
    "20250422_171703.braidz",
    "20250102_152013.braidz",
    "20241121_151406.braidz",
    "20250421_174810.braidz",
    "20250425_195715.braidz",
    "20250410_120718.braidz",
    "20250428_171938.braidz",
]


# Files for second strain Native
native = [
    "20250410_120718.braidz",
    "20250501_152448.braidz",
    "20250501_183152.braidz",
    "20250430_155345.braidz",
    "20241129_144233.braidz",
    "20250106_153631.braidz",
    "20241127_152044.braidz",
    "20241116_154109.braidz",
    "20250108_160315.braidz",
    "20241125_132912.braidz",
    "20241114_173118.braidz",
    "20250427_170153.braidz",
    "20241112_124059.braidz",
]


# In[4]:


SCREEN2HEADING_DATA = """
screen,heading
0,2.3513283485530456
80,1.2179812647799937
160,0.5031545295746856
240,-0.3078141744904855
320,-0.8746949393526915
400,-1.5019022477483523
480,-2.185375561680841
560,-3.0123437340031307
640,2.3513283485530456
"""
get_heading_func = create_interpolation_function()


# In[5]:


# Setup paths
braidz_path = "/gpfs/soma_fs/nfc/nfc3008/Experiments/"
braidz_files = glob.glob(os.path.join(braidz_path, "*.braidz"))

slp_path = "/gpfs/soma_fs/home/buchsbaum/sleap_projects/highspeed/predictions/"
slp_folders = glob.glob(
    os.path.join(
        slp_path,
        "*",
    )
)
slp_folders = natsorted(slp_folders)
output_path = "/gpfs/soma_fs/home/buchsbaum/src/sleap_video_analysis/output"


# In[6]:


# Loop over slp files and find the corresponding braidz files
for slp_folder in tqdm(slp_folders):
    filename = pathlib.Path(slp_folder).stem
    date = filename.split("_")[0]

    braidz_file_to_search = os.path.join(braidz_path, filename + ".braidz")

    # check if braidz file exists in `braidz_files` list
    braidz_file = [f for f in braidz_files if filename in f]
    braidz_file = braidz_file[0] if braidz_file else None
    if braidz_file is None:
        print(f"No braidz file found for {filename}, trying to search for another file with date {date}")

        # if braidz file not found, search for braidz files with the same date but a different time
        braidz_file = [f for f in braidz_files if date in f]
        braidz_file = braidz_file[0] if braidz_file else None

        if braidz_file is None:
            print(f"No braidz file found for {filename}, skipping")
            continue

    # make sure `braidz_file` exists
    if not os.path.exists(braidz_file):
        print(f"Braidz file {braidz_file} does not exist, skipping")
        continue

    # extract stimulus information from braidz file
    process_braidz_file(braidz_file, "data", get_heading_func)


# # Helper functions

# In[7]:


def calculate_heading_difference(a1, a2):
    # Calculate the angular difference considering the circular nature
    diff = a1 - a2

    # Normalize to [-π, π] range
    return np.arctan2(np.sin(diff), np.cos(diff))


def sg_smooth(array, window_length=51, polyorder=3, **kwargs):
    return savgol_filter(
        array, window_length=window_length, polyorder=polyorder, **kwargs
    )


def unwrap_with_nan(array):
    array[~np.isnan(array)] = np.unwrap(array[~np.isnan(array)])
    return array


def detect_tracking_gaps(df, min_tracked_frames=10, min_gap_size=20):
    """
    Detects if there are multiple tracking sections separated by NaN gaps in the data.

    Parameters:
        df (pd.DataFrame): The dataframe with tracking data (complete_df)
        min_tracked_frames (int): Minimum consecutive frames to consider a valid tracking section
        min_gap_size (int): Minimum size of NaN gap to alert about

    Returns:
        bool: True if multiple tracking sections with gaps are detected
    """
    # Create a mask for rows where all tracking points are valid
    valid_mask = (
        ~pd.isna(df["head.x"])
        & ~pd.isna(df["head.y"])
        & ~pd.isna(df["abdomen.x"])
        & ~pd.isna(df["abdomen.y"])
    )

    # Convert mask to integers (1 for valid, 0 for NaN)
    valid_series = valid_mask.astype(int)

    # Detect changes in the mask (0->1 or 1->0)
    # This creates a series where 1 indicates the start or end of a tracking section
    changes = valid_series.diff().abs()

    # Get indices where changes occur
    change_indices = np.where(changes == 1)[0]

    # If less than 2 changes, there's only one section or no valid sections
    if len(change_indices) < 2:
        return False

    # Calculate segments
    segments = []

    # If the first frame is valid, the first change is the end of a segment
    start_idx = 0 if valid_series.iloc[0] == 1 else change_indices[0]

    for i in range(1 if valid_series.iloc[0] == 1 else 2, len(change_indices), 2):
        if i >= len(change_indices):
            # If we have an odd number of changes and started with a valid segment
            end_idx = len(valid_series) - 1
        else:
            end_idx = change_indices[i] - 1

        # Only include segments that are long enough
        segment_length = end_idx - start_idx + 1
        if segment_length >= min_tracked_frames:
            segments.append((start_idx, end_idx, segment_length))

        # Set up for next segment if there are more changes
        if i + 1 < len(change_indices):
            start_idx = change_indices[i + 1]

    # If we have only one valid segment, no need to alert
    if len(segments) <= 1:
        return False

    # Check gaps between segments
    for i in range(len(segments) - 1):
        current_end = segments[i][1]
        next_start = segments[i + 1][0]
        gap_size = next_start - current_end - 1

        if gap_size >= min_gap_size:
            # print(f"ALERT: Multiple tracking sections detected!")
            # print(f"  Section 1: Frames {segments[i][0]}-{segments[i][1]} ({segments[i][2]} frames)")
            # print(f"  Gap: {gap_size} frames with NaNs")
            # print(f"  Section 2: Frames {segments[i+1][0]}-{segments[i+1][1]} ({segments[i+1][2]} frames)")
            return True

    return False


def savgol_filter_with_nans(y, window_length, polyorder, **kwargs):
    """
    Apply savgol_filter to an array that contains NaNs.
    The filter is only applied to contiguous segments of non-NaN values.

    Parameters:
    -----------
    y : array_like
        The data to be filtered
    window_length : int
        The length of the filter window (must be odd)
    polyorder : int
        The order of the polynomial used to fit the samples
    **kwargs : dict
        Additional arguments to pass to savgol_filter

    Returns:
    --------
    y_filtered : ndarray
        The filtered data with NaNs preserved in their original locations
    """
    # Create a copy of the input array to avoid modifying the original
    y_filtered = np.copy(y)

    # Find indices of non-NaN values
    valid_indices = ~np.isnan(y)

    if not np.any(valid_indices):
        return y_filtered  # Return original if all values are NaN

    # Find contiguous segments of valid data
    diff_indices = np.diff(np.concatenate(([0], valid_indices.astype(int), [0])))
    start_indices = np.where(diff_indices == 1)[0]
    end_indices = np.where(diff_indices == -1)[0]
    segments = zip(start_indices, end_indices)

    # Apply savgol_filter to each segment separately
    for start, end in segments:
        # Only apply filter if the segment is long enough
        if end - start >= window_length:
            y_filtered[start:end] = savgol_filter(
                y[start:end], window_length, polyorder, **kwargs
            )
        # Leave shorter segments unfiltered

    return y_filtered


# In[8]:


def process_data(stim_csvs_folder, pre_range=[0, 400], post_range=[400, 750]):
    """
    This function accepts a folder with all the csv files that contain the stimulus data as
    extracted from the braid recording.
    Then, for each file, it finds the correct folder with the converted slp files, and
    inside that folder finds the correct file that matches each row (obj_id + frame) in the stim csv file.

    It then loads the data from that file, and calculates the heading difference between pre and post
    stimulus data, as well as the heading difference between pre stimulus and the stimulus heading.
    The results are returned as a pandas DataFrame.

    Parameters:
        stim_csvs_folder (str): The folder containing the stimulus CSV files.
        pre_range (list): The range of frames to consider for pre-stimulus data.
        post_range (list): The range of frames to consider for post-stimulus data.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data with heading differences.
    """
    # Create an empty list to collect all the data
    all_data = []

    # get all csv files in the stim_csvs_folder (these are the stim files)
    stim_csvs = sorted(glob.glob(os.path.join(stim_csvs_folder, "*.csv")))

    # define pattern recognition for filenames
    pattern = r"obj_id_(\d+)_frame_(\d+)"

    # loop over all files
    for stim_csv in stim_csvs:
        date = datetime.strptime(os.path.basename(stim_csv), "%Y%m%d_%H%M%S.csv").date()

        # check if date is before 1st of April 2025
        if date < datetime(2025, 4, 1).date():
            print(f"Skipping file {stim_csv} because it's before April 1st, 2025")
            continue

        print(f"==== Processing {stim_csv} ====")
        stim_df = pd.read_csv(stim_csv)  # read the csv

        # now get the correct folder for the stim file
        slp2csv_folder = os.path.join(
            stim_csvs_folder,
            os.path.join(
                os.path.basename(os.path.normpath(stim_csv)).replace(".csv", "")
            ),
        )

        # and get all the files from that folder
        slp2csv_files = sorted(glob.glob(os.path.join(slp2csv_folder, "*.csv")))
        if len(slp2csv_files) == 0:
            print(f"No slp2csv files found in {slp2csv_folder}, skipping")
            continue

        # loop over the rows of each stim file
        for idx, row in stim_df.iterrows():
            # extract data for each stim row
            stim_obj_id = int(row["obj_id"])
            stim_frame = int(row["frame"])
            stim_heading = float(row["stim_heading"])

            # Find the matching csv file
            matching_file = None
            for file in slp2csv_files:
                match = re.search(pattern, file)
                if match:
                    file_obj_id = int(match.group(1))
                    file_frame = int(match.group(2))

                    if file_obj_id == stim_obj_id and file_frame == stim_frame:
                        matching_file = file
                        break

            # if no matching file was found, skip
            if matching_file is None:
                continue

            # Load the matching file
            data_df = pd.read_csv(matching_file)

            # Check if the original data has too few tracked frames
            if len(data_df) < 51:
                # print(f"Skipping file with insufficient data: {matching_file}")
                continue

            # Create an empty DataFrame with the same structure as data_df
            complete_df = pd.DataFrame(columns=data_df.columns)

            # Set dtypes to match the original dataframe
            for col in data_df.columns:
                complete_df[col] = complete_df[col].astype(data_df[col].dtype)

            # Fill the frame_idx column with all possible frames (0-749)
            complete_df["frame_idx"] = np.array(range(750))

            # Set the index to frame_idx for easier merging
            complete_df = complete_df.set_index("frame_idx")
            data_df_indexed = data_df.set_index("frame_idx")

            # Update the complete_df with values from the original data_df
            complete_df.update(data_df_indexed)

            # Reset index to make frame_idx a column again
            complete_df = complete_df.reset_index()

            # Example usage in your code:
            if detect_tracking_gaps(
                complete_df, min_tracked_frames=10, min_gap_size=20
            ):
                has_tracking_gaps = True

            # Now interpolate to fill the gaps in tracking data
            data_df_interp = complete_df.interpolate(
                method="linear", limit_direction="both", limit=25
            )

            # extract all data and apply smoothing
            frames = data_df_interp["frame_idx"].to_numpy()
            head_x = savgol_filter_with_nans(
                data_df_interp["head.x"].to_numpy(), window_length=51, polyorder=3
            )
            head_y = savgol_filter_with_nans(
                data_df_interp["head.y"].to_numpy(), window_length=51, polyorder=3
            )
            abdomen_x = savgol_filter_with_nans(
                data_df_interp["abdomen.x"].to_numpy(), window_length=51, polyorder=3
            )
            abdomen_y = savgol_filter_with_nans(
                data_df_interp["abdomen.y"].to_numpy(), window_length=51, polyorder=3
            )

            # head_x = sg_smooth(data_df_interp["head.x"].to_numpy())
            # head_y = sg_smooth(data_df_interp["head.y"].to_numpy())
            # abdomen_x = sg_smooth(data_df_interp["abdomen.x"].to_numpy())
            # abdomen_y = sg_smooth(data_df_interp["abdomen.y"].to_numpy())

            # extract all frames in pre and post stimulus ranges
            pre_indices = np.where((frames >= pre_range[0]) & (frames < pre_range[1]))[
                0
            ]
            post_indices = np.where(
                (frames >= post_range[0]) & (frames < post_range[1])
            )[0]

            # Count frames with valid (non-NaN) tracking data in pre-range
            pre_valid_mask = (
                ~np.isnan(head_x[pre_indices])
                & ~np.isnan(head_y[pre_indices])
                & ~np.isnan(abdomen_x[pre_indices])
                & ~np.isnan(abdomen_y[pre_indices])
            )
            pre_valid_count = np.sum(pre_valid_mask)

            # Count frames with valid (non-NaN) tracking data in post-range
            post_valid_mask = (
                ~np.isnan(head_x[post_indices])
                & ~np.isnan(head_y[post_indices])
                & ~np.isnan(abdomen_x[post_indices])
                & ~np.isnan(abdomen_y[post_indices])
            )
            post_valid_count = np.sum(post_valid_mask)

            # Skip if not enough valid frames in these ranges
            if pre_valid_count < 10 or post_valid_count < 10:
                continue

            # Keep only valid indices
            pre_indices = pre_indices[pre_valid_mask]
            post_indices = post_indices[post_valid_mask]

            # extract pre-stimulus coordinates
            head_x_pre = head_x[pre_indices]
            head_y_pre = head_y[pre_indices]
            abdomen_x_pre = abdomen_x[pre_indices]
            abdomen_y_pre = abdomen_y[pre_indices]

            # extract post-stimulus coordinates
            head_x_post = head_x[post_indices]
            head_y_post = head_y[post_indices]
            abdomen_x_post = abdomen_x[post_indices]
            abdomen_y_post = abdomen_y[post_indices]

            # calculate heading for each frame (angle of vector from abdomen to head)
            pre_heading = np.arctan2(
                head_y_pre - abdomen_y_pre, head_x_pre - abdomen_x_pre
            )
            post_heading = np.arctan2(
                head_y_post - abdomen_y_post, head_x_post - abdomen_x_post
            )

            # calculate circular mean of headings (accounts for circular nature of angle data)
            pre_heading_mean = circmean(pre_heading, high=np.pi, low=-np.pi)
            post_heading_mean = circmean(post_heading, high=np.pi, low=-np.pi)

            # calculate heading differences
            try:
                # Calculate the difference between post-stimulus and pre-stimulus headings
                prepost_heading_difference = calculate_heading_difference(
                    post_heading_mean, pre_heading_mean
                )

                # Calculate the difference between stimulus heading and pre-stimulus heading
                prestim_heading_difference = calculate_heading_difference(
                    stim_heading, pre_heading_mean
                )

                # Calculate the difference between stimulus heading and post-stimulus heading
                poststim_heading_difference = calculate_heading_difference(
                    stim_heading, post_heading_mean
                )

                # Determine if the fly turned toward or away from the stimulus
                # Get sign of stimulus position relative to fly
                stimulus_direction = np.sign(prestim_heading_difference)

                # Get sign of the turn
                turn_direction = np.sign(prepost_heading_difference)

                # If stimulus_direction and turn_direction have opposite signs,
                # the fly turned away from the stimulus
                turned_away = stimulus_direction * turn_direction < 0
                # Extract the base filename without any extension
                basename = os.path.splitext(os.path.basename(stim_csv))[0]

                # Create sets of basenames without extensions (more efficient for lookups)
                native_basenames = {os.path.splitext(filename)[0] for filename in native}
                canton_s_basenames = {os.path.splitext(filename)[0] for filename in canton_s}
                
                row_data = row.to_dict()  # Convert the row to a dictionary
                # Determine the group
                if basename in native_basenames:
                    row_data["group"] = "Native"
                elif basename in canton_s_basenames:
                    row_data["group"] = "Canton-s"
                else:
                    row_data["group"] = "Unknown"  # Handle case where it doesn't match either list
                
                row_data["obj_id"] = int(row_data["obj_id"])
                row_data["frame"] = int(row_data["frame"])
                row_data["file"] = basename
                row_data["prepost_heading_difference"] = prepost_heading_difference
                row_data["prestim_heading_difference"] = prestim_heading_difference
                row_data["poststim_heading_difference"] = poststim_heading_difference
                row_data["pre_heading"] = pre_heading_mean
                row_data["post_heading"] = post_heading_mean
                row_data["turned_away"] = turned_away
                row_data["turn_direction"] = "Away" if turned_away else "Toward"

                # Append to the all_data list
                all_data.append(row_data)
            
            except ValueError as e:
                print(f"Error calculating heading difference: {e}")
                continue
    # Create a DataFrame from all the collected data
    result_df = pd.DataFrame(all_data)

    # Now result_df contains all the data from all files with the heading differences
    print(f"Combined DataFrame has {len(result_df)} rows")
    return result_df


# In[9]:


results_df = process_data(
    "/gpfs/soma_fs/home/buchsbaum/src/sleap_video_analysis/data"
)


# In[10]:


# bin data by `prestim_heading_difference`
# where -45 to +45 is `front`, -45 to -135 is `left`, +45 to +135 is `right`, rest is 'back'
def bin_heading_difference(row):
    if (
        row["prestim_heading_difference"] >= -np.pi / 4
        and row["prestim_heading_difference"] <= np.pi / 4
    ):
        return "front"
    elif (
        row["prestim_heading_difference"] > np.pi / 4
        and row["prestim_heading_difference"] <= 3 * np.pi / 4
    ):
        return "right"
    elif (
        row["prestim_heading_difference"] < -np.pi / 4
        and row["prestim_heading_difference"] >= -3 * np.pi / 4
    ):
        return "left"
    else:
        return "back"


# Apply the function to create a new column
results_df["prestim_heading_bin"] = results_df.apply(
    bin_heading_difference, axis=1
)


# In[11]:


def plot_histogram_for_all_bins(results_df, label):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    bins = results_df["prestim_heading_bin"].unique()
    for i, bin in enumerate(bins):
        ax = axs[i // 2, i % 2]
        bin_df = results_df[results_df["prestim_heading_bin"] == bin]
        sns.histplot(
            data=bin_df,
            x="prepost_heading_difference",
            bins=np.linspace(-np.pi, np.pi, 30),
            common_bins=True,
            stat="density",
            label=bin,
            ax=ax,
            common_norm=True,
            kde=True,
        )

        # add line for median and mean
        mean = circmean(bin_df["prepost_heading_difference"], high=np.pi, low=-np.pi)

        ax.axvline(mean, color="g", linestyle="--", label="Mean")
        ax.set_title(f"Heading Difference Histogram - {bin}")

    fig.suptitle(label, fontsize=16)
    plt.tight_layout()
    plt.show()


plot_histogram_for_all_bins(results_df[results_df["group"] == "Canton-s"], "CS")
plot_histogram_for_all_bins(results_df[results_df["group"] == "Native"], "Native")


# # Clustering

# In[146]:


def get_data_for_clustering(stim_csvs_folder, window=25, mean=False,):
    # Create a list to store all feature matrices
    all_features = []
    # Create a list to store metadata
    all_metadata = []

    # get all csv files in the stim_csvs_folder (these are the stim files)
    stim_csvs = sorted(glob.glob(os.path.join(stim_csvs_folder, "*.csv")))

    # define pattern recognition for filenames
    pattern = r"obj_id_(\d+)_frame_(\d+)"
    raw_data = {}
    # Inside your nested loops, after extracting features:
    for i, stim_csv in enumerate(stim_csvs):
        print(f"Processing file {stim_csv} ({i} out of {len(stim_csvs)})")
        # print(f"==== Processing {stim_csv} ====")
        stim_df = pd.read_csv(stim_csv)  # read the csv

        # now get the correct folder for the stim file
        slp2csv_folder = os.path.join(
            stim_csvs_folder,
            os.path.join(
                os.path.basename(os.path.normpath(stim_csv)).replace(".csv", "")
            ),
        )

        # and get all the files from that folder
        slp2csv_files = sorted(glob.glob(os.path.join(slp2csv_folder, "*.csv")))

        for j, (idx, row) in enumerate(stim_df.iterrows()):
            # extract data for each stim row
            stim_obj_id = int(row["obj_id"])
            stim_frame = int(row["frame"])
            stim_heading = float(row["stim_heading"])

            # Find the matching csv file
            matching_file = None
            for file in slp2csv_files:
                match = re.search(pattern, file)
                if match:
                    file_obj_id = int(match.group(1))
                    file_frame = int(match.group(2))

                    if file_obj_id == stim_obj_id and file_frame == stim_frame:
                        matching_file = file
                        break

            # if no matching file was found, skip
            if matching_file is None:
                continue

            # Load the matching file
            data_df = pd.read_csv(matching_file)

            # Create an empty DataFrame with the same structure as data_df
            complete_df = pd.DataFrame(columns=data_df.columns)

            # Set dtypes to match the original dataframe
            for col in data_df.columns:
                complete_df[col] = complete_df[col].astype(data_df[col].dtype)

            # Fill the frame_idx column with all possible frames (0-749)
            complete_df["frame_idx"] = list(range(750))

            # Set the index to frame_idx for easier merging
            complete_df = complete_df.set_index("frame_idx")
            data_df_indexed = data_df.set_index("frame_idx")

            # Update the complete_df with values from the original data_df
            complete_df.update(data_df_indexed)

            # Reset index to make frame_idx a column again
            complete_df = complete_df.reset_index()

            # Example usage in your code:

            # Now interpolate to fill the gaps in tracking data
            data_df_interp = complete_df.interpolate(
                method="linear", limit_direction="both", limit=25
            )

            # extract all data and apply smoothing
            frames = data_df_interp["frame_idx"].to_numpy()
            head_x = savgol_filter_with_nans(
                data_df_interp["head.x"].to_numpy(), window_length=51, polyorder=3
            )
            head_y = savgol_filter_with_nans(
                data_df_interp["head.y"].to_numpy(), window_length=51, polyorder=3
            )
            abdomen_x = savgol_filter_with_nans(
                data_df_interp["abdomen.x"].to_numpy(), window_length=51, polyorder=3
            )
            abdomen_y = savgol_filter_with_nans(
                data_df_interp["abdomen.y"].to_numpy(), window_length=51, polyorder=3
            )

            # Calculate different features
            heading = np.arctan2(head_y - abdomen_y, head_x - abdomen_x)
            heading_unwrap = unwrap_with_nan(heading)
            heading_change = savgol_filter(
                np.gradient(heading_unwrap, 1 / 500), window_length=21, polyorder=3
            )
            distance_between_head_and_abdomen = np.sqrt(
                (head_x - abdomen_x) ** 2 + (head_y - abdomen_y) ** 2
            )

            # calculate centroid based on head and abodomen coordinates
            centroid_x = (head_x + abdomen_x) / 2
            centroid_y = (head_y + abdomen_y) / 2
            centroid_velocity_x = savgol_filter_with_nans(
                np.gradient(centroid_x, 1 / 500), window_length=21, polyorder=3
            )
            centroid_velocity_y = savgol_filter_with_nans(
                np.gradient(centroid_y, 1 / 500), window_length=21, polyorder=3
            )
            centroid_velocity = np.sqrt(centroid_velocity_x**2 + centroid_velocity_y**2)
            centroid_acceleration = savgol_filter_with_nans(
                np.gradient(centroid_velocity, 1 / 500), window_length=21, polyorder=3
            )

            theta = np.arctan2(centroid_velocity_y, centroid_velocity_x)
            theta_unwrap = unwrap_with_nan(theta)
            angular_velocity = savgol_filter_with_nans(
                np.gradient(theta_unwrap, 1 / 500), window_length=21, polyorder=3
            )

            # find peaks in the angular velocity
            positive_peaks, _ = find_peaks(
                angular_velocity, height=np.deg2rad(1000), distance=50
            )
            negative_peaks, _ = find_peaks(
                -angular_velocity, height=np.deg2rad(1000), distance=50
            )
            all_peaks = np.sort(np.concatenate((positive_peaks, negative_peaks)))

            # find if there are any peaks in range
            response_peak = [peak for peak in all_peaks if 350 < peak < 450]

            # get response_peak
            if len(response_peak) == 0:
                response_peak = 390 # this is the average response time based on previous calculations
            else:
                response_peak = response_peak[0]

            # get window to extract
            if window == 1:
                indices = response_peak
            else:
                indices = range(response_peak - window, response_peak + window + 1)

            # extract the window
            heading = heading[indices]

            # ignoring nans
            if np.any(np.isnan(heading)):
                continue
            
            # extract all other parameters
            heading_change = heading_change[indices]
            distance_between_head_and_abdomen = distance_between_head_and_abdomen[
                indices
            ]
            centroid_velocity = centroid_velocity[indices]
            centroid_acceleration = centroid_acceleration[indices]
            angular_velocity = angular_velocity[indices]

            frames = frames[indices]
            x = centroid_x[indices]
            y = centroid_y[indices]

            # Extract the base filename without any extension
            basename = os.path.splitext(os.path.basename(stim_csv))[0]

            # Create sets of basenames without extensions (more efficient for lookups)
            native_basenames = {os.path.splitext(filename)[0] for filename in native}
            canton_s_basenames = {os.path.splitext(filename)[0] for filename in canton_s}
            
            # Determine the group name first
            if basename in native_basenames:
                group_name = "Native"
            elif basename in canton_s_basenames:
                group_name = "Canton-s"
            else:
                group_name = "Unknown"
            
            # save all raw data
            raw_data[os.path.basename(matching_file)] = {
                "obj_id": int(row["obj_id"]),
                "frame": int(row["frame"]),
                "stim_heading": float(row["stim_heading"]),
                "group": group_name,
                "stim_csv": os.path.basename(stim_csv),
                "slp_csv": os.path.basename(matching_file),
                "stimulus_position": stim_heading,
                "response_peak": response_peak,
                "heading": heading,
                "heading_change": heading_change,
                "distance_between_head_and_abdomen": distance_between_head_and_abdomen,
                "centroid_velocity": centroid_velocity,
                "centroid_acceleration": centroid_acceleration,
                "angular_velocity": angular_velocity,
                "x": x,
                "y": y,
                "frames": frames,
            }

            # Apply as single value or list based on window condition
            if window == 1 or mean:
                group = group_name
                file = os.path.basename(matching_file)
                
                frames = np.nanmean(frames)
                x = np.nanmean(x)
                y = np.nanmean(y)

                heading_sd = np.nanstd(heading)
                heading = np.nanmean(heading)

                heading_change_sd = np.nanstd(heading_change)
                heading_change = np.nanmean(heading_change)

                distance_between_head_and_abdomen_sd = np.nanstd(
                    distance_between_head_and_abdomen
                )
                distance_between_head_and_abdomen = np.nanmean(
                    distance_between_head_and_abdomen
                )

                centroid_velocity = np.nanmean(centroid_velocity)
                centroid_velocity_sd = np.nanstd(centroid_velocity)

                centroid_acceleration_sd = np.nanstd(centroid_acceleration)
                centroid_acceleration = np.nanmean(centroid_acceleration)
                
                angular_velocity_sd = np.nanstd(angular_velocity)
                angular_velocity = np.nanmean(angular_velocity)


                response_peak = np.nanmean(response_peak)
                
            else:
                group = [group_name] * len(indices)
                file = [os.path.basename(matching_file)] * len(indices)

            temp_dict = {
                "file": file,
                "group": group,
                "time": frames,
                "x": x,
                "y": y,
                "heading": heading,
                "heading_sd": heading_sd,
                "heading_change": heading_change,
                "heading_change_sd": heading_change_sd,
                "distance_between_head_and_abdomen": distance_between_head_and_abdomen,
                "distance_between_head_and_abdomen_sd": distance_between_head_and_abdomen_sd,
                "centroid_velocity": centroid_velocity,
                "centroid_velocity_sd": centroid_velocity_sd,
                "centroid_acceleration": centroid_acceleration,
                "centroid_acceleration_sd": centroid_acceleration_sd,
                "angular_velocity": angular_velocity,
                "angular_velocity_sd": angular_velocity_sd,
                "response_delay": response_peak,
                
            }

            if window == 1 or mean:
                all_features.append(temp_dict)
            else:
                all_features.append(pd.DataFrame(temp_dict))
    
    return pd.DataFrame(all_features) if (window == 1 or mean) else pd.concat(all_features, ignore_index=True), raw_data


# In[147]:


all_features, raw_data = get_data_for_clustering(
    "/gpfs/soma_fs/home/buchsbaum/src/sleap_video_analysis/data/", window=25, mean=True,
)


# In[103]:


print(f"NaN before interpolation: {all_features.isna().sum().sum()}")

# Print which columns have NaN and how many per column
nan_counts = all_features.isna().sum()
columns_with_nan = nan_counts[nan_counts > 0]
print("Columns with NaN values:")
if len(columns_with_nan) > 0:
    for column, count in columns_with_nan.items():
        print(f"  {column}: {count}")
else:
    print("  None")

interp_features = all_features.interpolate()
print(f"NaN after interpolation: {interp_features.isna().sum().sum()}")


# In[104]:


from sklearn.preprocessing import StandardScaler

# Extract just the numeric features
numeric_features = interp_features[['heading', 'heading_sd', 
                                    'heading_change', 'heading_change_sd',
                                    'distance_between_head_and_abdomen', 'distance_between_head_and_abdomen_sd',
                                    'centroid_velocity', 'centroid_velocity_sd',
                                    'centroid_acceleration', 'centroid_acceleration_sd',
                                    'angular_velocity', 'angular_velocity_sd',]]

# Standardize features
features_scaled = StandardScaler().fit_transform(numeric_features)


# In[105]:


features_scaled.shape


# In[ ]:





# In[121]:


import umap
from itertools import product
# Define parameter grid
n_neighbors_list = [10, 15, 30, 50]
min_dist_list = [0.0, 0.1, 0.25, 0.5]
metrics = ['euclidean', 'correlation', 'cosine']

for (n_neighbors, min_dist, metric) in product(
    n_neighbors_list, min_dist_list, metrics
):
    print(f"n_neighbors: {n_neighbors}, min_dist: {min_dist}, metric: {metric}")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=42
    )

    embedding = reducer.fit_transform(features_scaled)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=interp_features["group"].astype("category").cat.codes,
        cmap="viridis",
        alpha=0.5,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP Projection - n_neighbors: {n_neighbors}, min_dist: {min_dist}, metric: {metric}")
    plt.colorbar(scatter, label="Group")
    plt.show()
    # Save the figure
    plt.savefig(f"./umap/{n_neighbors}_{min_dist}_{metric}.png")
    plt.close()


# In[125]:


n_neighbours = 30
min_dist = 0.1
method = "correlation"

reducer = umap.UMAP(
    n_neighbors=n_neighbours,
    min_dist=min_dist,
    n_components=2,
    metric=method,
    random_state=42
)

embedding = reducer.fit_transform(features_scaled)


# In[135]:


# Run HDBSCAN on the UMAP embedding
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
clusterer.fit(embedding)

# Get the cluster labels
labels = clusterer.labels_

# plot cluster with labeled groups
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=labels,
    cmap="Set1",
    alpha=0.5,
)


# In[144]:


# find which files are in each cluster
clusters = {}
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(all_features.iloc[i]["file"])


# In[167]:


cluster = []
angular_velocity = []
for key, value in clusters.items():
    for file in value:
        cluster.append(key)
        angular_velocity.append(raw_data[file]["angular_velocity"])

cluster = np.array(cluster)
angular_velocity = np.array(angular_velocity)


# In[168]:


def get_mean_and_std(array):
    mean = np.nanmean(array, axis=0)
    std = np.nanstd(array, axis=0)
    return mean, std

fig = plt.figure()
mean1, std1 = get_mean_and_std(angular_velocity[cluster == -1])
mean2, std2 = get_mean_and_std(angular_velocity[cluster == 0])
mean3, std3 = get_mean_and_std(angular_velocity[cluster == 1])
mean4, std4 = get_mean_and_std(angular_velocity[cluster == 2])

plt.plot(mean1, label="Cluster -1", color="red")
plt.fill_between(
    range(len(mean1)), mean1 - std1, mean1 + std1, color="red", alpha=0.2
)

plt.plot(mean2, label="Cluster 0", color="blue")
plt.fill_between(
    range(len(mean2)), mean2 - std2, mean2 + std2, color="blue", alpha=0.2
)

plt.plot(mean3, label="Cluster 1", color="green")
plt.fill_between(
    range(len(mean3)), mean3 - std3, mean3 + std3, color="green", alpha=0.2
)

plt.plot(mean4, label="Cluster 2", color="orange")
plt.fill_between(
    range(len(mean4)), mean4 - std4, mean4 + std4, color="orange", alpha=0.2
)

plt.title("Angular Velocity for each cluster")
plt.xlabel("Frame")
plt.ylabel("Angular Velocity")
plt.legend()
plt.show()


# In[ ]:




