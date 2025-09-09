
import os
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Load the averaged DataFrame
TO_ANALYZE = pd.read_csv(os.path.join(PROCESSED_DIR, "connectivity_combined_patient.csv"))

regions = [
    'frontal', 'left_frontal_temporal', 'right_frontal_temporal',
    'left_frontal_parietal', 'right_frontal_parietal',
    'left_frontal_within', 'right_frontal_within'
]

# Assuming df_averaged has 'depression' column indicating status
# Create a DataFrame with depression status (assuming it's in merged_df)
# We need to include this column in the averaged DataFrame first
# This assumes you have merged the original DataFrame with depression status before averaging
# If not, you can create a new column in df_averaged for demonstration purposes

# DROP BAD PATIENTS IF NECESSARY

# Separate groups based on depression status
depressed = TO_ANALYZE[TO_ANALYZE['depression'] == 1]
not_depressed = TO_ANALYZE[TO_ANALYZE['depression'] == 0]

# List of features to test
features = [f'{i+1}_epoch_{region}' for region in regions for i in range(30)]

# Perform t-tests for each feature
results = {}
for feature in features:
    t_stat, p_value = stats.ttest_ind(depressed[feature], not_depressed[feature], nan_policy='omit')
    results[feature] = {'t_stat': t_stat, 'p_value': p_value}

# Create a DataFrame to display results
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(PROCESSED_DIR, "t_test_results.csv"), index=True)

print(results_df)

#                                t_stat   p_value
# frontal_mean                 1.080232  0.284669
# left_frontal_temporal_mean   0.327438  0.744558
# right_frontal_temporal_mean  0.894335  0.374970
# left_frontal_parietal_mean   1.113904  0.270079
# right_frontal_parietal_mean  1.417863  0.161771
# left_frontal_within_mean     0.094445  0.925092
# right_frontal_within_mean   -0.112934  0.910487
# frontal_std                 -1.185699  0.240748
# left_frontal_temporal_std    0.036995  0.970620
# right_frontal_temporal_std  -1.330135  0.188866
# left_frontal_parietal_std   -0.183806  0.854830
# right_frontal_parietal_std  -0.696411  0.489052
# left_frontal_within_std     -0.031314  0.975131
# right_frontal_within_std     0.199598  0.842518