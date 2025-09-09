
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

folder_path_connectivity = "C:/Users/fmava/Downloads/EEG Learning/MDD Dataset Practice/Files"



print("Concatenating ND array...")
connectivity_data = []
for filename in os.listdir(folder_path_connectivity):

    if filename.endswith("_PLI.nc"):

        # Important info
        patient_id = filename[:3]
        file_path = os.path.join(folder_path_connectivity, filename)

        # Read connectivity data using MNE
        connectivity = mne_connectivity.read_connectivity(file_path)

        # Extract connectivity data as a matrix as a NumPy array
            # NOTE: by default ‘raveled’, which will represent each connectivity matrix as a (n_nodes_in * n_nodes_out,)
            # list. If ‘dense’, then will return each connectivity matrix as a 2D array. If ‘compact’ (default) then will
            # return ‘raveled’ if indices were defined as a list of tuples, or dense if indices is ‘all’. Multivariate
            # connectivity data cannot be returned in a dense form.
            # https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.Connectivity.html#mne_connectivity.Connectivity.get_data

        conn_data = connectivity.get_data(output="raveled").mean(axis=1)
        print(conn_data.shape)

        # NEED TO CONVERT COMPLEX NUMBERS TO REAL NUMBERS -> magnitude is taken
        # conn_data = np.abs(conn_data)

        to_add = [patient_id] + conn_data.tolist()
        # Append to list
        connectivity_data.append(to_add)

        # print(f"Shape of {patient_id} connectivity: {conn_data.shape}")
        # print(len(connectivity.names))

print("Done creating ND array")
# connectivity: 3844, 10
# 62 nodes total

# ------------------------ CREATE TWO SEPARATE DATA FRAMES ------------------------
# ONE WHERE EACH ROW CORRESPONDS TO A SPECIFIC REGION, ANOTHER WHERE ALL THE REGIONS ARE COMBINED INTO A SINGLE ROW
print('Starting row regions calculation\n')

df_separate = pd.DataFrame(connectivity_data)
df_separate.to_csv("C:/Users/fmava/Downloads/EEG Learning/MDD Dataset Practice/connectivity_pairwise.csv")
df_separate.columns = ['patient_id'] + [f'col_{i}' for i in range(1, df_separate.shape[1])]
df_filtered = df_separate.loc[:, (df_separate != 0).any(axis=0)]
df_filtered.columns = ['patient_id'] + [f'col_{i}' for i in range(1, df_filtered.shape[1])]
df_filtered.to_csv("C:/Users/fmava/Downloads/EEG Learning/MDD Dataset Practice/connectivity_pairwise_filtered.csv")

print("Filtered:\n", df_filtered[:10])

#   0     1     2     3     4     5     6     ...      3838      3839      3840      3841      3842      3843  3844
# 0  512   0.0   0.0   0.0   0.0   0.0   0.0  ...  0.421804  0.419888  0.412566  0.417934  0.397641  0.393819   0.0
# 1  514   0.0   0.0   0.0   0.0   0.0   0.0  ...  0.389225  0.401719  0.400659  0.387414  0.393062  0.399372   0.0
# 2  518   0.0   0.0   0.0   0.0   0.0   0.0  ...  0.399604  0.442689  0.406082  0.400627  0.394748  0.390258   0.0
# 3  519   0.0   0.0   0.0   0.0   0.0   0.0  ...  0.403003  0.386890  0.374792  0.388799  0.372669  0.401974   0.0
# 4  520   0.0   0.0   0.0   0.0   0.0   0.0  ...  0.406044  0.417824  0.404888  0.411696  0.402173  0.410115   0.0
# 5  522   0.0   0.0   0.0   0.0   0.0   0.0  ...  0.421665  0.408797  0.395288  0.409254  0.392392  0.411382   0.0
# 6  523   0.0   0.0   0.0   0.0   0.0   0.0  ...  0.418236  0.428822  0.457216  0.435328  0.412126  0.421799   0.0
# 7  524   0.0   0.0   0.0   0.0   0.0   0.0  ...  0.407433  0.429881  0.434864  0.445710  0.425985  0.414737   0.0
