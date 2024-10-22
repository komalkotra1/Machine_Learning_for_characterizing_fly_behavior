import os
import shutil

# Define paths
csv_folder = 'C:/Users/kotra/Documents/Videos/20230801_134131/tracking'  # Folder containing the original CSV files
tracking_folder = 'C:/Users/kotra/Documents/Videos/20230801_134131/trajectory'  # Folder containing the tracking files
output_folder = 'C:/Users/kotra/Documents/Videos/20230801_134131/tracking_saccades'  # Folder to save matched CSV files

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the CSV folder (whether CSV or not) to debug
all_files_in_csv_folder = os.listdir(csv_folder)
print(f"All files in {csv_folder}: {all_files_in_csv_folder}")

# Get the list of CSV file names (without extensions)
csv_files = {f[:-4]: f for f in os.listdir(csv_folder) if f.lower().endswith('.csv')}
print(f"CSV files (with extensions): {csv_files}")

# Image extensions to check for (in tracking_folder)
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

# Use a set to track copied CSV files
copied_csv_files = set()

# Set to track processed base names to avoid double copying
processed_base_names = set()

# Loop through all files in the tracking folder
for file in os.listdir(tracking_folder):
    file_name, file_ext = os.path.splitext(file)

    # Check if the file is an image with a valid extension
    if file_ext.lower() in image_extensions and '_trajectory_plot' in file_name:
        # Extract the base name (before "_trajectory_plot")
        base_name = file_name.replace('_trajectory_plot', '')

        # Ensure that the base name hasn't been processed before
        if base_name in processed_base_names:
            print(f"Skipping base name {base_name} as it has already been processed.")
            continue

        # Check if there's a corresponding CSV file in the csv_folder
        if base_name in csv_files:
            # Get the full CSV filename and paths
            csv_file = csv_files[base_name]

            # Debug: Print the base name and CSV file to be copied
            print(f"Found matching base name: {base_name} -> {csv_file}")

            # Only copy the CSV if it hasn't been copied yet
            if csv_file not in copied_csv_files:
                source_path = os.path.join(csv_folder, csv_file)
                destination_path = os.path.join(output_folder, csv_file)

                # Copy the CSV file to the output folder
                shutil.copy(source_path, destination_path)
                copied_csv_files.add(csv_file)  # Mark this CSV file as copied
                processed_base_names.add(base_name)  # Mark this base name as processed
                print(f"Copied {csv_file} to {output_folder}")
            else:
                print(f"CSV file {csv_file} has already been copied.")
        else:
            print(f"No CSV file found for base name: {base_name}")
    else:
        print(f"Skipping {file}: Not a valid tracking file or doesn't match pattern")

print("\nAll corresponding CSV files have been processed.")
