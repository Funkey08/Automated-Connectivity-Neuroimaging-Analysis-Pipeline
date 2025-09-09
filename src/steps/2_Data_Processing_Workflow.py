# This is a data processing workflow for individuals with all 67 channels

import os
import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat # Not entirely sure how to import the raw .mat data
from scipy.special import gamma
import mne_connectivity
from mne_connectivity import spectral_connectivity_time
from mne_connectivity.viz import plot_sensors_connectivity
from mne_connectivity import read_connectivity
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator
from mne.viz import circular_layout
from math import log
import tkinter as tk
from tkinter import simpledialog
from typing import List, Dict, Optional

import autoreject

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

csv_file = os.path.join(PROCESSED_DIR, "patients_with_meg.csv")
entry_to_start = 13
number_of_entries_to_access = 10

# Read the CSV file to get the list of patients with MEG channels
df = pd.read_csv(csv_file)  # Assuming the file has a single column with filenames
print(df)


def preprocess_patient_data(epochs, patient_file, file_number, output_folder):

    print(f'\nNOW PREPROCESSING PATIENT {patient_file}\n')

    # FIRST AUTOREJECT
    ar = autoreject.AutoReject(random_state=42)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    evoked_bad = epochs[reject_log.bad_epochs].average() if epochs[reject_log.bad_epochs] else None

    # FIRST ICA
    ica = mne.preprocessing.ICA(max_iter='auto', random_state=42)
    ica.fit(epochs[~reject_log.bad_epochs])
    print(f'ICA explains {ica.get_explained_variance_ratio(filtered_data)} of variance')

    ecg_indices, ecg_scores = ica.find_bads_ecg(filtered_data)
    # print(ecg_indices)
    # ica.plot_scores(ecg_scores);

    eog_indices, eog_scores = ica.find_bads_eog(filtered_data)
    # print(eog_indices)
    # ica.plot_scores(eog_scores);

    # AUTO REMOVE ECG AND EOG ICA COMPONENTS
    ica.exclude = eog_indices + ecg_indices

    # APPLY ICA
    ica.apply(epochs, exclude=ica.exclude)

    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    epochs_ar = ar.transform(epochs, reject_log=reject_log)
    print(f'Number of epochs in patient {file_number} originally: {len(epochs)}, '
          f'after autoreject: {len(epochs_ar)}')

    output_file_path = os.path.join(output_folder, 'Files', f'{file_number}_Depression_REST_CLEANED-epo.fif')
    epochs_ar.save(output_file_path, overwrite=True)

    print(f'\nDone preprocessing {patient_file}!!!\n')


def get_connectivity_measures(methods: List[str],
                              min_freq: int,
                              max_freq: int,
                              freqs: List[int],
                              eeg_epochs,
                              return_connectivity: bool=False,
                              overwrite: bool=False
                              ) -> Dict[str, np.ndarray]:
    # con_ = {} -> can add this if we want to save connectivity data as well
    con_data = {}
    for method in methods:
        connectivity_file_path = os.path.join(PROCESSED_DIR, 'Files',
                                           f'{file_number}_Depression_REST_{method.upper()}')

        if not os.path.exists(connectivity_file_path) or overwrite:
            action = "OVERWRITING" if overwrite else "NO"
            print(f"\n{action} {method.upper()} connectivity for patient {file_number}, calculating now...\n")
            connectivity = spectral_connectivity_time(eeg_epochs, freqs, method=method, average=True,
                                                      mode='multitaper', fmin=min_freq, fmax=max_freq)
            connectivity.save(connectivity_file_path)
        else:
            print(f'\n{method.upper()} connectivity for patient {file_number} already exists, retrieving data now\n')
            connectivity = read_connectivity(connectivity_file_path)

        con_data[method] = connectivity.get_data(output='dense')

        return con_data if return_connectivity else None



def get_canonical_coherence_measures(indices,
                                     min_freq: int,
                                     max_freq: int,
                                     freqs: List[int],
                                     eeg_epochs,
                                     n_cycles: List[float],
                                     file_marker: str,
                                     return_connectivity: bool=False,
                                     overwrite: bool=False
                                     ) -> Optional[mne_connectivity.Connectivity]:

    # channel names: ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
    # 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    # 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
    # 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2', 'HEOG',
    # 'VEOG','EKG']

    connectivity_file_path = os.path.join(PROCESSED_DIR, 'Files',
                                          f'{file_number}_Depression_REST_' + file_marker + '_CACOH')

    compute_new = not os.path.exists(connectivity_file_path) or overwrite
    action = "OVERWRITING" if overwrite else "NO" if compute_new else "RETRIEVING"

    print(f'\n{action} {file_marker} CANONICAL COHERENCE for patient {file_number}\n')

    if compute_new:
        connectivity = spectral_connectivity_time(eeg_epochs, freqs, indices=indices, method='cacoh', average=True,
                                                  mode='multitaper', n_cycles=n_cycles,
                                                  fmin=min_freq, fmax=max_freq)
        connectivity.save(connectivity_file_path)
    else:
        connectivity = read_connectivity(connectivity_file_path)

    return connectivity.get_data(output='dense') if return_connectivity else None


def get_specific_channel_names(channel_names: List[str], start: str, end: Optional[str] = None) -> List[str]:
    start_index = channel_names.index(start)
    if end is not None:
        end_index = channel_names.index(end) + 1
        return channel_names[start_index:end_index]
    else:
        return [channel_names[start_index]]


def get_specific_channel_indices(channel_names: List[str], start: str, end: Optional[str] = None) -> List[int]:
    start_index = channel_names.index(start)
    if end is not None:
        end_index = channel_names.index(end) + 1
        return list(range(start_index, end_index))
    else:
        return [start_index]



# -------------------- Main Processing --------------------

entry_to_start = int(input("Type row number of patient to start with:"))
number_of_entries_to_access = int(input("Type number of entries to access:"))


for patient_file in df['Filename'][entry_to_start:(entry_to_start + number_of_entries_to_access)]:

    print(patient_file)

    file_name = os.path.join(RAW_DIR, patient_file)
    file_number = patient_file[0:3]
    print(f'\nBeginning with patient number {file_number}...\n')
    mat_data = loadmat(file_name)
    eeg_signal = mat_data['EEG']['data'][0,0]
    sampling_rate =  mat_data['EEG']['srate'][0,0][0,0]
    chanlocs = mat_data['EEG']['chanlocs'][0,0]
    channel_names = [str(chan[0]) for chan in chanlocs['labels'][0]]

    special_mapping = {
        'HEOG': 'eog',
        'VEOG': 'eog',
        'EKG': 'ecg',
        'CB1': 'misc',
        'CB2': 'misc'
    }

    ch_types = [special_mapping.get(ch, 'eeg') for ch in channel_names]

    info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types=ch_types)

    info.set_montage("standard_1020", match_case=False, on_missing='warn')

    raw = mne.io.RawArray(eeg_signal, info)

    filtered_data = raw.copy().filter(l_freq=1, h_freq=None)


    # CREATE EOG AND ECG EVOKED
    eog_evoked = mne.preprocessing.create_eog_epochs(filtered_data).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))
    # eog_evoked.plot_joint(title=f'{patient_file}')
    ecg_evoked = mne.preprocessing.create_ecg_epochs(filtered_data).average()
    ecg_evoked.apply_baseline(baseline=(None, -0.2))
    # ecg_evoked.plot_joint();

    epochs = mne.make_fixed_length_epochs(filtered_data, duration=3, preload=True)

    # PREPROCESSING

    epochs_path = os.path.join(PROCESSED_DIR, 'Files', f'{file_number}_Depression_REST_CLEANED-epo.fif')
    if os.path.exists(epochs_path):
        print(f'\nPatient {file_number} already exists, retrieving data now\n')
        # Can enable plotting if user wants to
        # epochs_ar.plot(scalings='auto')
    else:
        preprocess_patient_data(epochs, patient_file, file_number, PROCESSED_DIR)
        
    epochs_ar = mne.read_epochs(epochs_path)

    eeg_epochs = epochs_ar.load_data().copy().pick('eeg')

    # ----- COMPUTE SIMPLE PAIRWISE CONNECTIVITY MEASURES -----
    # CHANGE DEPENDING ON WHAT TO STUDY
    min_freq = 4
    max_freq = 7
    freqs = np.linspace(min_freq, max_freq, 10)
    methods = ['pli', 'wpli', 'coh']

    con_data = get_connectivity_measures(methods, min_freq, max_freq, freqs, eeg_epochs, return_connectivity=True)

    # channel names: ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
    # 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    # 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
    # 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2', 'HEOG',
    # 'VEOG','EKG']

    # Optional: Include parietal occipital (PO) indices, FC indices, and temporal parietal (TP)?

    # ----- COMPUTE CANONICAL COHERENCE MULTIVARIATE CONNECTIVITY BETWEEN DIFFERENT REGIONS -----

    frontal_indices = ([get_specific_channel_indices(channel_names, 'F7', 'F1')],
                       [get_specific_channel_indices(channel_names, 'F2', 'F8')])

    left_frontal_temporal_indices = ([get_specific_channel_indices(channel_names, 'F7', 'F1')],
                                     [get_specific_channel_indices(channel_names, 'T7') +
                                     get_specific_channel_indices(channel_names, 'FT7') +
                                     get_specific_channel_indices(channel_names, 'TP7')])

    right_frontal_temporal_indices = ([get_specific_channel_indices(channel_names, 'F2', 'F8')],
                                     [get_specific_channel_indices(channel_names, 'T8') +
                                      get_specific_channel_indices(channel_names, 'FT8') +
                                      get_specific_channel_indices(channel_names, 'TP8')])

    left_frontal_parietal_indices = ([get_specific_channel_indices(channel_names, 'F7', 'F1')],
                                     [get_specific_channel_indices(channel_names, 'P7', 'P1')])

    right_frontal_parietal_indices = ([get_specific_channel_indices(channel_names, 'F2', 'F8')],
                                     [get_specific_channel_indices(channel_names, 'P2', 'P8')])

    left_frontal_within_indices = ([get_specific_channel_indices(channel_names, 'F7', 'F5')],
                                   [get_specific_channel_indices(channel_names, 'F3', 'F1')])

    right_frontal_within_indices = ([get_specific_channel_indices(channel_names, 'F2', 'F4')],
                                   [get_specific_channel_indices(channel_names, 'F6', 'F8')])

    INDICES_OF_INTEREST = [frontal_indices, left_frontal_temporal_indices, right_frontal_temporal_indices,
                           left_frontal_parietal_indices, right_frontal_parietal_indices,
                           left_frontal_within_indices, right_frontal_within_indices]

    FILE_CACOH_NAMING = ['frontal', 'left_frontal_temporal', 'right_frontal_temporal',
                         'left_frontal_parietal', 'right_frontal_parietal',
                         'left_frontal_within', 'right_frontal_within']

    # INDICES_OF_INTEREST = [frontal_indices, left_frontal_temporal_indices]
    #
    # FILE_CACOH_NAMING = ['frontal', 'left_frontal_temporal']
    
    min_freq = 1 # 1 to 40 possibly to remove slow drifts
    max_freq = 40 # possibly 40
    freqs = np.linspace(min_freq, max_freq, 30)
    # set time window to 2 seconds
    T_window = 2
    n_cycles = np.maximum(T_window * freqs, 1)

    for indices, file_name in zip(INDICES_OF_INTEREST, FILE_CACOH_NAMING):
        get_canonical_coherence_measures(indices, min_freq, max_freq, freqs, eeg_epochs, n_cycles,
                                         file_name, return_connectivity=False, overwrite=True)
