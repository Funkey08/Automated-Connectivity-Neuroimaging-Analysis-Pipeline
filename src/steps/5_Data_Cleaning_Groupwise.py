
# Activate MNE environment

# TO DO:
# how to deal with complex numbers -> made them absolute
# multiple regions of interest per patient (7) -> can either combine them into a single row, or keep them separate.
#   -> create two arrays to test them out

import os
import re
import numpy as np
import pandas as pd
import mne_connectivity

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

folder_path_connectivity = os.path.join(PROCESSED_DIR, 'Files')
bdi_file = None # NAME HERE

regions = [
    'frontal', 'left_frontal_temporal', 'right_frontal_temporal',
    'left_frontal_parietal', 'right_frontal_parietal',
    'left_frontal_within', 'right_frontal_within'
]

region_regex_pattern = r"(" + "|".join(regions) + ")"

connectivity_data = []

print("Concatenating ND array...")
for filename in os.listdir(folder_path_connectivity):

    if filename.endswith("CACOH.nc"):

        # Important info
        patient_id = filename[:3]
        file_path = os.path.join(folder_path_connectivity, filename)

        # Use regex to find the region in the filename
        match = re.search(region_regex_pattern, filename)
        if match:
            region = match.group(0)
        else:
            region = "UNKNOWN"

        # Read connectivity data using MNE
        connectivity = mne_connectivity.read_connectivity(file_path)

        # Extract connectivity data as a matrix as a NumPy array
            # NOTE: by default ‘raveled’, which will represent each connectivity matrix as a (n_nodes_in * n_nodes_out,)
            # list. If ‘dense’, then will return each connectivity matrix as a 2D array. If ‘compact’ (default) then will
            # return ‘raveled’ if indices were defined as a list of tuples, or dense if indices is ‘all’. Multivariate
            # connectivity data cannot be returned in a dense form.
            # https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.Connectivity.html#mne_connectivity.Connectivity.get_data

        conn_data = connectivity.get_data(output="raveled")

        # NEED TO CONVERT COMPLEX NUMBERS TO REAL NUMBERS -> magnitude is taken
        conn_data = np.abs(conn_data)

        # Convert to 1D matrix
        conn_flat = conn_data.flatten()

        to_add = [patient_id, region] + conn_flat.tolist()
        # Append to list
        connectivity_data.append(to_add)

        # print(f"Shape of {patient_id} {region} connectivity: {conn_flat.shape}")

print("Done creating ND array")

# print(connectivity_data[:5])

# NOTE: without extension,7 rows are generated for each patient, one for each region of interest:
#
#   patient_id                                       connectivity
# 0        512  [0.9410045012506321, 0.9246484774009156, 0.928...
# 1        512  [0.747357629901778, 0.6513185760817091, 0.7068...
# 2        512  [0.8143945678759611, 0.8245806735357981, 0.873...
# 3        512  [0.9597148773506933, 0.9705378531355493, 0.972...
# 4        512  [0.7686944778601971, 0.7373828502565469, 0.758...
# 5        512  [0.9546796062854707, 0.9579764713997055, 0.966...
# 6        512  [0.9545529194873391, 0.9688878864539345, 0.973...
# 7        514  [0.9610553444968467, 0.9637336110616713, 0.963...
# 8        514  [0.8328875154487028, 0.7661324898785765, 0.780...
# 9        514  [0.9437134991890191, 0.9585131234304068, 0.965...
#
# I can either FLATTEN (combine) rows for each patient OR keep multiple rows per patient to analyze how different
# regions contribute separately

# ------------------------ CREATE TWO SEPARATE DATA FRAMES ------------------------
# ONE WHERE EACH ROW CORRESPONDS TO A SPECIFIC REGION, ANOTHER WHERE ALL THE REGIONS ARE COMBINED INTO A SINGLE ROW
print('Starting row regions calculation\n')

# DEFINE COLUMN NAMES
num_features = len(conn_flat)

columns = ['patient_id', 'region'] + [f'{i + 1}_epoch' for i in range(num_features)]

df_separate = pd.DataFrame(connectivity_data, columns=columns)
df_separate.to_csv(os.path.join(PROCESSED_DIR, "connectivity_separate.csv"), index=False)

print(df_separate[:10])

#   patient_id                  region   1_epoch   2_epoch   3_epoch  ...  26_epoch  27_epoch  28_epoch  29_epoch  30_epoch
# 0        512                 frontal  0.941005  0.924648  0.928359  ...  0.880296  0.900257  0.903647  0.871699  0.873570
# 1        512   left_frontal_parietal  0.747358  0.651319  0.706864  ...  0.684550  0.663297  0.644870  0.659450  0.624088
# 2        512   left_frontal_temporal  0.814395  0.824581  0.873683  ...  0.809497  0.779999  0.786598  0.794122  0.763799
# 3        512     left_frontal_within  0.959715  0.970538  0.972014  ...  0.951138  0.950324  0.950359  0.948615  0.938592
# 4        512  right_frontal_parietal  0.768694  0.737383  0.758304  ...  0.650316  0.664153  0.654002  0.657135  0.626173
# 5        512  right_frontal_temporal  0.954680  0.957976  0.966734  ...  0.940753  0.908896  0.935327  0.927878  0.912409
# 6        512    right_frontal_within  0.954553  0.968888  0.973669  ...  0.953188  0.957844  0.957963  0.951538  0.950699
# 7        514                 frontal  0.961055  0.963734  0.963677  ...  0.885246  0.880855  0.892321  0.880869  0.855616
# 8        514   left_frontal_parietal  0.832888  0.766132  0.780839  ...  0.648522  0.676930  0.660479  0.645606  0.652314


# ------------------------ FLATTENED VERSION ------------------------
print('Starting flattened regions calculation\n')

# new array for combined data per patient_id
combined_data = []

# BEGIN BY ITERATING THRU EACH UNIQUE PATIENT_ID
for patient_id in df_separate['patient_id'].unique():

    # filter for the current patient_id
    patient_rows = df_separate[df_separate['patient_id'] == patient_id]

    # initialize a list to hold the combined data for this patient
    patient_combined = [patient_id]

    # loop thru each region to extract the epoch data
    for region in regions:
        # filter for the current region
        region_rows = patient_rows[patient_rows['region'] == region]
        epoch_data = region_rows.iloc[:, 2:].values.flatten()  # Get all epoch columns as a flattened array
        patient_combined.extend(epoch_data)

    combined_data.append(patient_combined)

combined_columns = ['patient_id'] + [f'{i + 1}_epoch_{region}' for region in regions for i in range(num_features)]


df_combined = pd.DataFrame(combined_data, columns=combined_columns)

df_combined.to_csv(os.path.join(PROCESSED_DIR, "connectivity_combined.csv"), index=False)

print("FINALLY! SAVED CONNECTIVITY DATA!!!")
print(df_combined.head())

# LET'S GOOOOOOOOOOOOOOOOOOOO!!!!!!!!!!!
#
#   patient_id  1_epoch_frontal  ...  29_epoch_right_frontal_within  30_epoch_right_frontal_within
# 0        512         0.941005  ...                       0.951538                       0.950699
# 1        514         0.961055  ...                       0.875903                       0.867768
# 2        518         0.908754  ...                       0.622620                       0.633216
# 3        519         0.923809  ...                       0.879621                       0.859507
# 4        520         0.989167  ...                       0.907899                       0.882530


# ------------------------ AVERAGED EPOCHS VERSION [SEPARATE] ------------------------


df = df_separate.copy()
epoch_columns = df.columns[2:]
df['connectivity_mean'] = df[epoch_columns].mean(axis=1)
df['connectivity_std'] = df[epoch_columns].std(axis=1)
new_df = df[['patient_id', 'region', 'connectivity_mean', 'connectivity_std']]
new_df.to_csv(os.path.join(PROCESSED_DIR, "connectivity_average_separate.csv"), index=False)


# ------------------------ AVERAGED EPOCHS VERSION [COMBINED] ------------------------


# Used to reduce from 30 * 7 = 210 features per row to 14 (mean and standard deviation)
#
# df = df_separate.copy()
# epoch_columns = df.columns[2:]
# df['connectivity_mean'] = df[epoch_columns].mean(axis=1)
# df['connectivity_std'] = df[epoch_columns].std(axis=1)
# new_df = df[['patient_id', 'region', 'connectivity_mean', 'connectivity_std']]
# new_df.to_csv("C:/Users/fmava/Downloads/EEG Learning/MDD Dataset Practice/TEST.csv")
# print('Starting averaged epochs calculation\n')

combined_data = []

# BEGIN BY ITERATING THRU EACH UNIQUE PATIENT_ID
for patient_id in new_df['patient_id'].unique():

    # filter for the current patient_id
    patient_rows = new_df[new_df['patient_id'] == patient_id]

    # initialize a list to hold the combined data for this patient
    patient_combined = [patient_id]

    # loop thru each region to extract the epoch data
    for region in regions:
        # filter for the current region
        region_rows = patient_rows[patient_rows['region'] == region]
        epoch_data = region_rows.iloc[:, 2:].values.flatten()  # Get all epoch columns as a flattened array
        patient_combined.extend(epoch_data)

    combined_data.append(patient_combined)

combined_columns = ['patient_id']
for region in regions:
    combined_columns.append(f'{region}_mean')
    combined_columns.append(f'{region}_std')

df_averaged = pd.DataFrame(combined_data, columns=combined_columns)

print(df_averaged.head())

# Save to CSV
df_averaged.to_csv(os.path.join(PROCESSED_DIR, "connectivity_average_combined.csv"), index=False)



# ------------------------ RAW CONNECTIVITY + AVERAGE + STD VERSION ------------------------

import pandas as pd

# Load the CSV files
df_raw = pd.read_csv(os.path.join(PROCESSED_DIR, "connectivity_combined.csv"))
df_averaged = pd.read_csv(os.path.join(PROCESSED_DIR, "connectivity_average_combined.csv"), index=False)

# Merge on 'patient_id'
df_combined = pd.merge(df_raw, df_averaged, on='patient_id', how='outer')

# Save the merged DataFrame
df_combined.to_csv(os.path.join(PROCESSED_DIR, "connectivity_raw_and_average_combined.csv"), index=False)

print(df_combined.head())

