import os
import pandas as pd
import shutil

# Path to the CSV file with binned data
binned_data_csv = "C:/Users/kotra/Documents/stimulus_analysis/difference_heading_binned_data_20230714_151827.csv"

# Path to the folder containing the original CSV files
original_files_folder = "C:/Users/kotra/Documents/Videos/20230714_151827/tracking_saccades/"

# Path to the folder where the subfolders will be created
output_folder = "C:/Users/kotra/Documents/stimulus_analysis/organized_files_20230714_151827/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the binned data from the CSV file
binned_df = pd.read_csv(binned_data_csv)

# Iterate over the rows of the DataFrame
for index, row in binned_df.iterrows():
    file_name = row['File Name']  # Get the file name from the CSV
    bin_label = row['Bin']        # Get the corresponding bin label (e.g., '0-45°')

    # Create a subfolder for the bin label if it doesn't exist
    bin_folder = os.path.join(output_folder, bin_label)
    os.makedirs(bin_folder, exist_ok=True)

    # Source file path (from the original folder)
    src_file_path = os.path.join(original_files_folder, file_name)

    # Destination file path (in the appropriate bin subfolder)
    dest_file_path = os.path.join(bin_folder, file_name)

    # Move the file to the corresponding bin folder
    if os.path.exists(src_file_path):
        shutil.move(src_file_path, dest_file_path)
        print(f"Moved {file_name} to {bin_folder}")
    else:
        print(f"File {file_name} not found in {original_files_folder}")

print("Files organized into bin-labeled subfolders successfully.")
