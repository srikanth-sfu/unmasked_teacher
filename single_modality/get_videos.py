import pandas as pd
import glob
import os
import shutil
import string

# Specify the folder containing CSV files and file pattern
file_pattern = "video_splits/ucf*.csv"  # Update the path accordingly
csv_files = glob.glob(file_pattern)

# Initialize an empty list to store the data
data = []

# Loop
# Loop through each file and append its content to the list
for file in csv_files:
    # Read the CSV file into a DataFrame without headers
    df = pd.read_csv(file, header=None)
    # Convert the DataFrame to a list of lists and append to data
    data.extend(df.values.tolist())

videos = glob.glob('data/*/*')

# Define the source directory and destination directory
source_dir = "data"  # Replace with your source folder path
destination_dir = "ucf"  # Replace with your destination folder path

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Define the list of video IDs (without extension) to filter
video_ids = set([x[0] for x in data])
# Iterate through the source directory recursively
found = set()
allowed_chars = string.ascii_letters + string.digits + "-_.!"
#filtered = [(x,'v_'+ "".join([y for y in os.path.basename(x) if y in allowed_chars])) for x in videos if 'Sovereigns' in x]
#print(filtered)
for root, _, files in os.walk(source_dir):
    for file in files:
        # Extract the file name without extension
        file_id, ext = os.path.splitext(file)
        # Check if the file name matches any of the video IDs
        if file_id in video_ids:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_dir, file_id+ext)
            # Copy file to destination
            shutil.copy(source_path, destination_path)
            print(f"Copied: {source_path} -> {destination_path}")
            found.add(file_id)
        else:
            # Construct full paths
            file_id_dst = "".join([x for x in file_id if x in allowed_chars])
            if file_id_dst in video_ids:
                found.add(file_id_dst)
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_dir, file_id_dst+ext)
                # Copy file to destination
                shutil.copy(source_path, destination_path)
                print(f"Copied: {source_path} -> {destination_path}")

print((video_ids-found))
