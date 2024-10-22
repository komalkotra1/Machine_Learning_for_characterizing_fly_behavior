import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file with binned data
binned_data_csv = "C:/Users/kotra/Documents/stimulus_analysis/20230801_134131_analysis/updated_binned_data_20230801_134131.csv"

# Read the binned data from the CSV file
binned_df = pd.read_csv(binned_data_csv)

# Group the data by 'Bin' column and count the number of flies (files) in each bin
grouped_data = binned_df.groupby('Bin').size()

# Create the plot
plt.figure(figsize=(10, 6))
grouped_data.plot(kind='bar', color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Bin (Degrees)')
plt.ylabel('Number of Flies (Files)')
plt.title('Number of Flies in Each Bin')

plt.tight_layout()
plt.savefig("C:/Users/kotra/Documents/stimulus_analysis/20230801_134131_analysis/number_of_flies_per_bin.png")

# Show or save the plot
plt.tight_layout()
plt.show()



