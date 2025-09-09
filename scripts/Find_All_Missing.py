import os
import glob
import scipy.io
import pandas as pd

# Path to your EEG dataset folder
data_folder = "C:/Users/fmava/Downloads/EEG Learning/MDD Dataset Practice/Matlab Files"

print('Program is running...')

# Define a function to check if MEG channels exist
def has_meg_channel(mat_file):
    """Check if the MAT file contains the required MEG channel."""
    try:
        mat_data = scipy.io.loadmat(mat_file)
        eeg_signal = mat_data['EEG']['data'][0, 0]
        sampling_rate = mat_data['EEG']['srate'][0, 0][0, 0]
        chanlocs = mat_data['EEG']['chanlocs'][0, 0]
        channel_names = [str(chan[0]) for chan in chanlocs['labels'][0]]

        # Replace 'meg_channel_name' with the actual key/name of the MEG channel in your .mat file
        return len(channel_names) == 67
    except Exception as e:
        print(f"Error loading {mat_file}: {e}")
        return False  # Skip files that can't be loaded

# Get all MAT files in the folder
all_files = glob.glob(os.path.join(data_folder, "*.mat"))

# Lists to store filenames
patients_with_meg = []
patients_missing_meg = []

# Loop through files and categorize them
for file_path in all_files:
    patient_id = os.path.basename(file_path)  # Get full filename

    if has_meg_channel(file_path):
        patients_with_meg.append(patient_id)
    else:
        patients_missing_meg.append(patient_id)

# Save lists to CSV
pd.DataFrame(patients_with_meg, columns=["Filename"]).to_csv("patients_with_meg.csv", index=False)
print('Patients with complete channels have been identified')
pd.DataFrame(patients_missing_meg, columns=["Filename"]).to_csv(os.path.join("patients_missing_meg.csv"), index=False)
print('Patients with missing channels have been identified')

print(f"Finished scanning! ✅ Found {len(patients_with_meg)} with MEG, {len(patients_missing_meg)} without.")
