#This i sthe first file of the extraction process run it first when you have downloaded dataset folder from your desired source
import tarfile
import os

# Define the name of the dataset file and the extraction directory
dataset_file = 'mvtec_anomaly_detection.tar.xz'
extraction_dir = 'dataset'

# Create the extraction directory if it doesn't exist
if not os.path.exists(extraction_dir):
    os.makedirs(extraction_dir)

# Extract the .tar.xz file
print(f"Extracting {dataset_file} to {extraction_dir}...")
with tarfile.open(dataset_file, 'r:xz') as tar:
    tar.extractall(path=extraction_dir)

print("Extraction complete.")
