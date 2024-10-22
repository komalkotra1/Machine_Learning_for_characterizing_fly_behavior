import numpy as np
import pandas as pd
import csv

calibration_file = 'C:/Users/kotra/Documents/stimulus_analysis/laser_movements.csv'
calibration_df = pd.read_csv(calibration_file)

def find_and_display_values(filename, object_id, frame):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename, sep=';', on_bad_lines='skip')

    # Check if the DataFrame is empty
    if df.empty:
        print("The DataFrame is empty. Please check the CSV file.")
        return

    # Print the original column names
    print("Original DataFrame columns:")
    print(df.columns)

    # Clean column names: strip whitespace and convert to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Debug: Print the cleaned column names
    print("Cleaned DataFrame columns:")
    print(df.columns)

    # Check if 'obj_id' and 'frame' exist in the DataFrame
    if 'obj_id' not in df.columns or 'frame' not in df.columns:
        print("Error: 'obj_id' or 'frame' column not found in the DataFrame.")
        return

    # Filter the DataFrame for matching object_id and frame
    matching_rows = df[(df['obj_id'] == object_id) & (df['frame'] == frame)]

    # Check if any matching rows were found
    if matching_rows.empty:
        print(f"No matching row found for object ID {object_id} and frame value {frame}.")
        return

    # Iterate through the matching rows (if there are multiple)
    for index, row in matching_rows.iterrows():
        try:
            # Accessing values from the columns directly using the row
            col2_value = row['obj_id']            # Get the object ID
            col3_value = row['frame']             # Get the frame number
            col4_value = row['x']                 # Get the x position
            col5_value = row['y']                 # Get the y position
            col23_value = row['looming_pos_x']    # Get the looming position x (assumed as column 23)

            # Display the extracted values
            print(f"Values from matching row (object ID: {object_id}, frame: {frame}):")
            print(f"Column 2 (Object ID): {col2_value}")
            print(f"Column 3 (Frame): {col3_value}")
            print(f"Column 4 (X): {col4_value}")
            print(f"Column 5 (Y): {col5_value}")
            print(f"Column 23 (Looming Position X): {col23_value}")
        except KeyError as e:
            print(f"Error accessing columns: {e}")

# Example usage
find_and_display_values("C:/Users/kotra/Documents/stimulus_analysis/20230626_161309.csv", 37861, 1315384)


def get_xy_array(filename):
    # Load the CSV file with the correct delimiter
    df = pd.read_csv(filename, delimiter=';')  # Try changing the delimiter here if it's not a comma

    # Standardize column names (lowercase and strip whitespace)
    df.columns = df.columns.str.strip().str.lower()
    print("Columns:", df.columns)

    # Check if 'x' and 'y' exist after correctly loading the columns
    try:
        x_column = df['x']  # Replace 'x' with the actual column name if necessary
        y_column = df['y']  # Replace 'y' with the actual column name if necessary
    except KeyError as e:
        print(f"Error accessing columns: {e}")
        return None  # Return None if columns are not found

    # Optional: Print some values for verification
    print("Sample X values:", x_column.head())
    print("Sample Y values:", y_column.head())

    # Returning x and y values as arrays or lists
    return x_column.values, y_column.values


# Test the function with the provided file path
result = get_xy_array(
    "c:/Users/kotra/Documents/Videos/20230626_161309/tracking/1_obj_id_33646_cam_23047980_frame_1129843.csv")

if result is not None:
    x_values, y_values = result
    # Optional: Print the returned arrays for verification
    print("X Values:", x_values)
    print("Y Values:", y_values)
else:
    print("Failed to retrieve x and y values.")

# xvel = np.diff(x_values)
# yvel = np.diff(y_values)

heading_before = np.arctan2(np.diff(y_values)[450:500],
                            np.diff(x_values)[450:500])  # this gives you the vector before in radians
# heading_after = np.arctan2(yvel[600:650], xvel[600:650]) #this gives you the vector after in radians
fly_heading_in_videos = heading_before
camera_number = 23047980  # from the video files
stim_position = 178  # from xpos

heading = calibration_df[calibration_df.camera == camera_number].heading_direction
screen = calibration_df[calibration_df.camera == camera_number].stim_direction
interp_heading = np.interp(stim_position, screen,  heading, period=2 * np.pi)
difference_in_heading = fly_heading_in_videos - interp_heading #ttheta f- theta stimulus
