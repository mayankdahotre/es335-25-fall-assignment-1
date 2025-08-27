import os
import urllib.request
import zipfile
import shutil

# Create Datasets folder in parent directory
parent_dir = os.path.dirname(os.getcwd())
datasets_dir = os.path.join(parent_dir, "Datasets")
os.makedirs(datasets_dir, exist_ok=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
zip_path = os.path.join(datasets_dir, "UCI_HAR_Dataset.zip")
temp_extract_path = os.path.join(datasets_dir, "temp_extract")
final_dataset_path = os.path.join(datasets_dir, "UCI HAR Dataset")

# Download the dataset
urllib.request.urlretrieve(url, zip_path)

# Extract to temporary folder
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(temp_extract_path)

# Move the UCI HAR Dataset folder to the correct location
extracted_dataset_path = os.path.join(temp_extract_path, "UCI HAR Dataset")
if os.path.exists(extracted_dataset_path):
    # Remove existing dataset if it exists
    if os.path.exists(final_dataset_path):
        shutil.rmtree(final_dataset_path)
    # Move the dataset to the final location
    shutil.move(extracted_dataset_path, final_dataset_path)

# Clean up temporary files and folders
if os.path.exists(temp_extract_path):
    shutil.rmtree(temp_extract_path)
if os.path.exists(zip_path):
    os.remove(zip_path)

