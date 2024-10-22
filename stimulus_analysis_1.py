import os
import numpy as np
import pandas as pd
import re
import os
import matplotlib as npl
import matplotlib.pyplot as plt

# Calibration file
calibration_file = 'C:/Users/kotra/Documents/stimulus_analysis/laser_movements.csv'
calibration_df = pd.read_csv(calibration_file)

# Folder containing CSV files
folder_path = "C:/Users/kotra/Documents/Videos/20230626_161309/tracking_saccades/"


# Function to extract camera number and frame number from the file name
def extract_camera_and_frame(file_name):
    # Use regex to find the camera number after 'cam_' and frame number after 'frame_'
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
        return None  # Skip this file if extraction fails

    # Load the CSV file with the correct delimiter
    df = pd.read_csv(file_path, delimiter=',', on_bad_lines='skip')

    # Standardize column names (lowercase and strip whitespace)
    df.columns = df.columns.str.strip().str.lower()

    # Check if 'x' and 'y' columns exist
    try:
        x_column = df['x']  # Replace 'x' with the actual column name if necessary
        y_column = df['y']  # Replace 'y' with the actual column name if necessary
    except KeyError as e:
        print(f"Error accessing 'x' or 'y' columns in {file_path}: {e}")
        return None

    # Calculate the velocity and heading
    x_values = x_column.values
    y_values = y_column.values
    heading_before = np.arctan2(np.diff(y_values)[450:500], np.diff(x_values)[450:500])
    fly_heading_in_videos = heading_before.mean()  # Taking the mean heading value over the range

    # Interpolate heading based on camera number
    camera_data = calibration_df[calibration_df.camera == camera_number]
    if camera_data.empty:
        print(f"No calibration data found for camera {camera_number} in {file_name}")
        return None

    # Extract the stim position corresponding to this frame
    stim_position = frame_number  # Assuming the frame number corresponds to the stimulus position
    heading = camera_data.heading_direction
    screen = camera_data.stim_direction
    interp_heading = np.interp(stim_position, screen, heading, period=2 * np.pi)

    # Calculate difference in heading
    difference_in_heading = fly_heading_in_videos - interp_heading

    # Return the result for this file
    return {
        "file_name": os.path.basename(file_path),
        "camera_number": camera_number,
        "frame_number": frame_number,
        "heading": fly_heading_in_videos,
        "screen": screen.mean() if not screen.empty else None,
        "interp_heading": interp_heading,
        "difference_in_heading": difference_in_heading
    }


# Collect results for all files
results = []

# Loop through all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        result = process_file(file_path, calibration_df)
        if result:
            results.append(result)
results
# Convert the results into a DataFrame and save to a CSV

results_df = pd.DataFrame(results)
results_df.to_csv("C:/Users/kotra/Documents/stimulus_analysis/20230626_161309_analysis/heading_analysis_results_20230714_151827.csv",
                  index=False)

print("Results saved successfully.")
diff_in_headings_eachfly = [result['difference_in_heading'] for result in results]

# Folder to save graphs
folder_graphs = 'C:/Users/kotra/Documents/stimulus_analysis/stimulus_analysis/diff_heading_graphs_20230626_161309'

# Create the folder if it doesn't exist
os.makedirs(folder_graphs, exist_ok=True)

# Iterate over the difference in heading for each file (fly) and plot the graph
for idx, difference_in_heading in enumerate(diff_in_headings_eachfly):
    # Convert the difference in heading to degrees
    difference_in_heading_deg = np.abs(difference_in_heading) * 57.29

    # Check if `difference_in_heading ` is a single value or an array
    if isinstance(difference_in_heading_deg, (np.ndarray, list)):
        # For multiple values, plot the time series
        instances = np.arange(1, len(difference_in_heading_deg) + 1)
        #plt.plot(instances, difference_in_heading_deg, label=f'Fly {idx + 1}', marker='o')+
        plt.bar(instances + idx * 0.2, difference_in_heading_deg, width=0.2, label=f'Fly {idx + 1}')
    else:
        # For single value, just plot one point
        plt.bar([idx + 1], [difference_in_heading_deg], label=f'Fly {idx + 1}')

# Add labels, title, and legend
plt.xlabel("Instances (or Flies)")
plt.ylabel("Difference in Heading (Degrees)")
plt.title("Difference in Heading for All Flies")
plt.legend()  # Shows which line corresponds to which fly

# Save the combined plot to file
plot_file_name = os.path.join(folder_graphs, "combined_diff_heading_all_flies_bar_with values_20230626_161309.png")
plt.savefig(plot_file_name)

# Show the combined plot
plt.show()

print("Combined graph plotted and saved successfully.")
for idx, difference_in_heading in enumerate(diff_in_headings_eachfly):
    difference_in_heading_deg = np.abs(difference_in_heading) * 57.29
    # Print values to the console
    if isinstance(difference_in_heading_deg, (np.ndarray, list)):
        for value in difference_in_heading_deg:
            print(f'Fly {idx + 1} Difference in Heading: {value:.2f}°')
    else:
        print(f'Fly {idx + 1} Difference in Heading: {difference_in_heading_deg:.2f}°')

print("Bar graph plotted and values printed successfully.")


