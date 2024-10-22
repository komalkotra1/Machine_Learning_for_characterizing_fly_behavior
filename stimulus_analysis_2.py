import os
import numpy as np
import pandas as pd
import re  # Importing the 're' module for regex operations

# Calibration file
calibration_file = 'C:/Users/kotra/Documents/stimulus_analysis/laser_movements.csv'
calibration_df = pd.read_csv(calibration_file)

# Folder containing CSV files
folder_path = "C:/Users/kotra/Documents/Videos/20230714_151827/tracking_saccades/"


# Function to extract camera number and frame number from the file name
def extract_camera_and_frame(file_name):
    camera_match = re.search(r'cam_(\d+)', file_name)
    frame_match = re.search(r'frame_(\d+)', file_name)

    if camera_match and frame_match:
        camera_number = int(camera_match.group(1))
        frame_number = int(frame_match.group(1))
        return camera_number, frame_number
    else:
        print(f"Could not extract camera and frame information from {file_name}")
        return None, None


# Function to process each file
def process_file(file_path, calibration_df):
    file_name = os.path.basename(file_path)

    # Extract camera number and frame number from the file name
    camera_number, frame_number = extract_camera_and_frame(file_name)
    if camera_number is None or frame_number is None:
        return None

    # Load the CSV file
    df = pd.read_csv(file_path, delimiter=',', on_bad_lines='skip')

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Check if 'x' and 'y' columns exist
    try:
        x_column = df['x']
        y_column = df['y']
    except KeyError as e:
        print(f"Error accessing 'x' or 'y' columns in {file_path}: {e}")
        return None

    # Calculate the velocity and heading
    x_values = x_column.values
    y_values = y_column.values

    # Ensure that the arrays have sufficient values before calculating heading
    if len(x_values) > 500 and len(y_values) > 500:
        heading_before = np.arctan2(np.diff(y_values)[450:500], np.diff(x_values)[450:500])

        # Check if heading_before is not empty
        if heading_before.size > 0:
            fly_heading_in_videos = heading_before.mean()
        else:
            print(f"Warning: Empty heading array in file {file_name}")
            return None
    else:
        print(f"Insufficient data in file {file_name} to calculate heading")
        return None

    # Interpolate heading based on camera number
    camera_data = calibration_df[calibration_df.camera == camera_number]
    if camera_data.empty:
        print(f"No calibration data found for camera {camera_number} in {file_name}")
        return None

    stim_position = frame_number
    heading = camera_data.heading_direction
    screen = camera_data.stim_direction
    interp_heading = np.interp(stim_position, screen, heading, period=2 * np.pi)

    # Calculate difference in heading
    difference_in_heading = fly_heading_in_videos - interp_heading

    # Return the result for this file
    return {
        "file_name": file_name,
        "difference_in_heading": np.abs(difference_in_heading) * 57.29  # Absolute difference in degrees
    }


# Collect results for all files
results = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        result = process_file(file_path, calibration_df)
        if result:
            results.append(result)

# Bin classification
angle_bins = [0, 45, 90, 135, 180]
binned_data = []

# Iterate over difference in headings and classify them into bins
for result in results:
    difference_in_heading_deg = result['difference_in_heading']

    # Ensure difference_in_heading_deg is iterable
    if isinstance(difference_in_heading_deg, (np.ndarray, list)):
        for angle in difference_in_heading_deg:
            # Clip angles to stay within 0-180 degrees
            angle = min(max(angle, 0), 180)
            # Determine the bin for each angle
            bin_idx = np.digitize(angle, angle_bins) - 1  # Adjusting for zero indexing
            # Ensure bin_idx does not exceed valid range
            if bin_idx < len(angle_bins) - 1:
                bin_label = f'{angle_bins[bin_idx]}-{angle_bins[bin_idx + 1]}°'
                binned_data.append({"Bin": bin_label, "Angle": angle, "File Name": result['file_name']})
    else:
        # Handle single values
        angle = min(max(difference_in_heading_deg, 0), 180)
        bin_idx = np.digitize(angle, angle_bins) - 1
        if bin_idx < len(angle_bins) - 1:
            bin_label = f'{angle_bins[bin_idx]}-{angle_bins[bin_idx + 1]}°'
            binned_data.append({"Bin": bin_label, "Angle": angle, "File Name": result['file_name']})

# Convert the binned data into a DataFrame
binned_df = pd.DataFrame(binned_data)

# Save the DataFrame to a CSV file
output_csv_path = "C:/Users/kotra/Documents/stimulus_analysis/difference_heading_binned_data_20230714_151827.csv"
binned_df.to_csv(output_csv_path, index=False)

print("Binned data saved successfully to", output_csv_path)
