
import os

folder_path = "C:/Users/fmava/Downloads/EEG Learning/MDD Dataset Practice/Files"

for filename in os.listdir(folder_path):
    if filename.endswith("PLI"):
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, filename + ".nc")
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {filename}.nc')

print("Renaming Complete")
