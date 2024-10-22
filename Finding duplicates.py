import os
import pandas as pd
import hashlib


def hash_file(file_path):
    """Calculate SHA256 hash for the file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def find_duplicate_files(folder_path):
    """Find and delete duplicate CSV files in the folder."""
    file_hashes = {}  # Dictionary to store file hashes
    files_to_delete = []  # List to keep track of duplicates

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):  # Check if it's a CSV file
            file_path = os.path.join(folder_path, filename)

            # Calculate file hash
            file_hash = hash_file(file_path)

            # Check if this hash already exists (duplicate file)
            if file_hash in file_hashes:
                print(f"Duplicate found: {filename} is a duplicate of {file_hashes[file_hash]}.")
                files_to_delete.append(file_path)  # Mark for deletion
            else:
                file_hashes[file_hash] = filename  # Store the hash with the filename

    # Delete the duplicate files
    for file_path in files_to_delete:
        os.remove(file_path)
        print(f"Deleted duplicate file: {file_path}")


if __name__ == "__main__":
    folder_path = input("C:/Users/kotra/Documents/Videos/20230801_134131/tracking_saccades")

    if os.path.exists(folder_path):
        find_duplicate_files(folder_path)
        print("Duplicate removal process completed.")
    else:
        print("The specified folder does not exist.")
