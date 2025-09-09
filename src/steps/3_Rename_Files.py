
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

folder_path = os.path.join(PROCESSED_DIR, 'Files')

for filename in os.listdir(folder_path):
    if filename.endswith("PLI"):
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, filename + ".nc")
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {filename}.nc')

print("Renaming Complete")
