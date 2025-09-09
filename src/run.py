
import os
import subprocess

# Mapping from menu choices to script filenames
SCRIPT_MAP = {
    "1": "1_Find_All_Missing.py",
    "2": "2_Data_Processing_Workflow.py",
    "3": "3_Rename_Files.py",
    "4": "4_Data_Cleaning_Pairwise.py",
    "5": "5_Data_Cleaning_Groupwise.py",
    "6": "6_ML_Analysis_Connectivity_Pairwise.py",
    "7": "7_ML_Analysis_Connectivity_Groupwise.py",
    "8": "8_ML_Analysis_Connectivity_Groupwise_Average.py",
    "9": "9_ML_Analysis_TTest.py",
}

MENU = """
=========================================
AUTOMATIC CONNECTIVITY ANALYSIS PIPELINE 
=========================================

Please select a step to begin. The steps listed are in recommended order.

1. Find All Missing
- Goes through the .MAT files in /data/raw. Reads them and checks to see if the MAT files contain the required MEG channels.
- Creates TWO CSVs: one for patients WITH MEG (patients_with_meg.csv) and one for patients WITHOUT MEG (patients_without_meg.csv).

2. Data Processing Workflow
- Runs AUTOREJECT, ICA, removes ECG & EOG components, recomputes ICA, and computes connectivity matrices.
- Produces a cleaned FIF, [pairwise method, default PLI], and CACOH connectivity file for each patient.

3. Rename Files
- Adds .nc extension to generated connectivity files. This should be run immediately affter step 2.

4. Data Cleaning Pairwise
- Combines pairwise files into ND matrices.
- Produces:
    • connectivity_pairwise.csv (raw pairwise)
    • connectivity_pairwise_filtered.csv (columns of zeroes removed to remove uninformative homogeneity)

5. Data Cleaning Groupwise
- Combines canonical coherence CACOH files into ND matrices.
- Produces:
    • connectivity_combined.csv
    • connectivity_separate.csv
    • connectivity_average_combined.csv
    • connectivity_average_separate.csv
    • connectivity_raw_and_average_combined.csv

6. ML Analysis Connectivity Pairwise
- Trains RF model on connectivity_pairwise_filtered.csv.
- Produces connectivity_pairwise_patient.csv and prints RF metrics.

7. ML Analysis Connectivity Groupwise
- Trains an optimized RF model on connectivity_combined.csv.
- Produces connectivity_combined_patient.csv and prints RF metrics.

8. ML Analysis Connectivity Groupwise Average
- Trains an optimized RF model on connectivity_average_combined.csv.
- Produces connectivity_averaged_combined_patient.csv and prints RF metrics.

9. ML Analysis T-Test
- Performs epoch-level T-tests on connectivity_combined_patient.csv.

Type a number (1-9) to run a step.
Type 'q' to quit.
"""

def run_script(choice: str):
    """Runs the selected script using subprocess."""
    script = os.path.join("steps",SCRIPT_MAP.get(choice))
    if script:
        print(f"\n>>> Running {script}...\n")
        try:
            subprocess.run(["python", script], check=True)
            print(f"\n✅ Finished running {script}!\n")
        except subprocess.CalledProcessError:
            print(f"\n❌ Error: Failed to run {script}\n")
    else:
        print("\n⚠️ Invalid selection. Please choose a valid step.\n")

def main():
    """Main interactive loop."""
    while True:
        print(MENU)
        choice = input("Select a step to run: ").strip()
        if choice.lower() == "q":
            print("\nExiting pipeline. Goodbye!\n")
            break
        run_script(choice)

if __name__ == "__main__":
    main()
