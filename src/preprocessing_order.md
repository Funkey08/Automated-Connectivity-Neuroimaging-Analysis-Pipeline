# Selectable Steps
It is highly recommended to run the steps in the shown order.


### 1. Find All Missing
- Goes through the .MAT files. Reads themand checks to see if the MAT files contains the required MEG channels (for heartbeat filtration). 
- Creates TWO CSVs: one containing all patients WITH MEG (patients_with_meg.csv) and all patients WITHOUT MEG (patients_without_meg.csv).

### 2. Data Processing Workflow
- Goes through all the files in patients_with_meg.csv and AUTOREJECTS, conducts ICA and AUTOREMOVES ECG AND ICA COMPONENTS, and does ICA again.
- SAVES the data as {file_number}_Depression_REST_CLEANED-epo.fif.
- Then gets simple pairwise connectivity AND canonical coherence multivariate measures between different brain regions. CAN PASS IN A LIST of methods in order to perform analysis on each one.
- Creates files for each, named appropriately

### 3. Rename Files
- Run immediately after producing PLI files; adds .nc extension if missing.

### 4. Data Cleaning Pairwise
- Finds all ..._PLI.nc files. 
- Creates two data frames as CSVs:
  - connectivity_pairwise.csv - the RAW pairwise data
  - connectivity_pairwise_filtered.csv - the pairwises data with columns of zeroes REMOVED

### 5. Data Cleaning Groupwise
- Finds all canonical coherence .nc files: ...CACOH.nc and creates a single ND dataframe.
- Then, creates CSVs:
  - connectivity_combined.csv (rows represent "patient", columns represent "epoch-region")
  - connectivity_separate.csv (basically a long pivot, so that rows represent "patient-region" and columns represent "epoch")
- As well as CSVs for averaged data:
  - connectivity_average_combined.csv (contains mean and std of each region)
  - connectivity_average_separate.csv (basically a long pivot)
- Creates one last CSV which merges the epoch and averaged data:
  - connectivity_raw_and_average_combined.csv (basically connectivity_combined.csv + connectivity_average_combined.csv)

### 6. ML Analysis Connectivity Pairwise
- Uses connectivity_pairwise_filtered.csv and the Depression_Dataset.csv to create connectivity_pairwise_patient.csv, which is used for analysis
- Conducts grid search + trains RF on connectivity_pairwise_patient.csv
  
### 7. ML Analysis Connectivity Groupwise
- Uses connectivity_combined.csv and the Depression_Dataset.csv to create connectivity_combined_patient.csv, which is used for analysis
- Conducts grid search + trains RF on connectivity_combined_patient.csv

### 8. ML Analysis Connectivity Average
- Uses connectivity_average_combined.csv and the Depression_Dataset.csv to create connectivity_average_patient.csv, which is used for analysis
- Conducts grid search + trains RF on connectivity_average_combined_patient.csv

### 9. ML Analysis TTest
- Runs epoch-level T-testing using connectivity_combined_patient.csv

### (Optional) NC_ML_Analysis
- Examines CACOH (Canonical Connectivity) files.
- Shows the top 10 most important features using an RF grid search.


## TLDR; CSVs produced by this pipeline:
- Find All Missing
  - patients_missing_meg.csv
  - patients_with_meg.csv
- Data Cleaning Pairwise
  - connectivity_pairwise.csv
  - connectivity_pairwise_filtered.csv
- Data Cleaning Groupwisee
  - connectivity_combined.csv
  - connectivity_separate.csv
  - connectivity_average_combined.csv
  - connectivity_average_separate.csv
  - connectivity_raw_and_average_combined.csv
- ML Analysis Connectivity Pairwise
  - connectivity_pairwise_patient.csv
- ML Analysis Connectivity Groupwise
  - connectivity_combined_patient.csv
- ML Analysis Connectivity Average
  - connectivity_average_combined_patient.csv
- ML Analysis TTest
  - t_test_results.csv
