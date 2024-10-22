import pandas as pd

# File paths
unbinned_file = "C:/Users/kotra/Documents/stimulus_analysis/20230801_134131_analysis/heading_analysis_results_20230801_134131.csv"
binned_file = "C:/Users/kotra/Documents/stimulus_analysis/20230801_134131_analysis/difference_heading_binned_data.csv"
output_file = "C:/Users/kotra/Documents/stimulus_analysis/20230801_134131_analysis/updated_binned_data_20230801_134131.csv"

# Load the unbinned file (containing 'difference_in_heading')
unbinned_df = pd.read_csv(unbinned_file)

# Load the binned file
binned_df = pd.read_csv(binned_file)

# Check the column names in both DataFrames to verify
print("Unbinned file columns:", unbinned_df.columns)
print("Binned file columns:", binned_df.columns)

# Since the binned file has 'File Name' instead of 'file_name', we'll match on 'File Name'
# Adjust 'File Name' to match the actual column names in your files if needed
binned_df['File Name'] = binned_df['File Name'].str.strip()  # Clean 'File Name' column in binned file

# Merge 'difference_in_heading' into the binned file based on 'File Name'
# Match on 'File Name' between the two DataFrames
binned_df = binned_df.merge(unbinned_df[['file_name', 'difference_in_heading']], left_on='File Name', right_on='file_name', how='left')

# Drop the extra 'file_name' column that was added during the merge
binned_df = binned_df.drop(columns=['file_name'])

# Save the updated binned file
binned_df.to_csv(output_file, index=False)

print(f"Updated binned file saved at: {output_file}")