import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to the CSV file with binned data
binned_data_csv = "C:/Users/kotra/Documents/stimulus_analysis/20230801_134131_analysis/updated_binned_data_20230801_134131.csv"

# Path to the folder containing the organized video subfolders
organized_videos_folder = "C:/Users/kotra/Documents/stimulus_analysis/20230801_134131_analysis/organized_videos"

# Read the binned data from the CSV file
binned_df = pd.read_csv(binned_data_csv)

# Group the data by 'Bin' column
grouped_data = binned_df.groupby('Bin')

# Iterate over each bin (subfolder)
for bin_label, group in grouped_data:
    # Get the fly headings for this bin
    headings = group['difference_in_heading'].values
    files = group['File Name'].values

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(headings, bins=np.arange(0, 181, 15), edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel('Difference in Heading (Degrees)')
    plt.ylabel('Number of Flies')
    plt.title(f'Difference in Heading for {bin_label}° Bin')

    # Save the plot inside the respective bin folder
    bin_folder = os.path.join(organized_videos_folder, bin_label)
    plot_file_name = os.path.join(bin_folder, f'heading_distribution_{bin_label.replace("°", "")}.png')
    plt.savefig(plot_file_name)
    plt.close()

    print(f"Graph saved for {bin_label} bin in {bin_folder}")

print("Graphs generated and saved successfully.")
