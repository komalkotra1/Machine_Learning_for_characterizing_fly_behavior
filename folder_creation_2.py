import os
import pandas as pd
import shutil

# Path to the CSV file with binned data
binned_data_csv = "C:/Users/kotra/Documents/stimulus_analysis/20230714_151827_analysis/difference_heading_binned_data_20230714_151827.csv"

# Path to the folder containing the video files
video_files_folder = "C:/Users/kotra/Downloads/Videos/20230714_151827/Flights"

# Path to the folder where the new organized video subfolders will be created
output_videos_folder = "C:/Users/kotra/Documents/stimulus_analysis/20230714_151827_analysis/organized_videos_20230714_151827"

# Create the output folder if it doesn't exist
os.makedirs(output_videos_folder, exist_ok=True)

# Read the binned data from the CSV file
binned_df = pd.read_csv(binned_data_csv)

# Iterate over the rows of the DataFrame
for index, row in binned_df.iterrows():
    file_name = row['File Name']  # Get the file name from the CSV
    bin_label = row['Bin']        # Get the corresponding bin label (e.g., '0-45°')

    # Create a subfolder for the bin label if it doesn't exist
    bin_folder = os.path.join(output_videos_folder, bin_label)
    os.makedirs(bin_folder, exist_ok=True)

    # The video file extension (change to the correct extension if needed)
    video_file_extension = '.mp4'  # Assuming videos are in .mp4 format

    # Source video file path (from the video folder)
    video_file_name = os.path.splitext(file_name)[0] + video_file_extension  # Change the extension to .mp4
    src_video_path = os.path.join(video_files_folder, video_file_name)

    # Destination video file path (in the appropriate bin subfolder)
    dest_video_path = os.path.join(bin_folder, video_file_name)

    # Move the video file to the corresponding bin folder
    if os.path.exists(src_video_path):
        shutil.move(src_video_path, dest_video_path)
        print(f"Moved {video_file_name} to {bin_folder}")
    else:
        print(f"Video file {video_file_name} not found in {video_files_folder}")

print("Video files organized into bin-labeled subfolders successfully.")
