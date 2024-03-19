#! G:\Internship\Surge\For Evaluation\Project\.venv\Scripts\python.exe

import os
import glob
import pandas as pd

def preprocess_datatables(data_directory, preprocessed_directory, columns_with_meaningful_nan=None):
    """
    Adjusted script with shortened column names for better visibility in terminals and editors with limited display width.
    """
    if columns_with_meaningful_nan is None:
        columns_with_meaningful_nan = {}

    # Ensure the preprocessed directory exists
    os.makedirs(preprocessed_directory, exist_ok=True)
    
    # List all CSV files in the data_directory
    csv_files = glob.glob(os.path.join(data_directory, '*.csv'))
    
    # Initialize lists to store summary information
    summary_before = []
    summary_after = []

    for file_path in csv_files:
        # Load the dataset, attempting to parse dates
        df = pd.read_csv(file_path, parse_dates=True, infer_datetime_format=True)
        df = df.drop_duplicates()
        
        # Replace NaN in specified columns with meaningful values
        for column, value in columns_with_meaningful_nan.items():
            if column in df.columns:
                df[column].fillna(value, inplace=True)

        # Initial summary before preprocessing
        initial_summary = {
            'Dataset name': os.path.basename(file_path),
            'Col Cnt': df.shape[1],
            'Row Cnt': df.shape[0],
            'Missing Col Cnt': df.isnull().any().sum(),
            'TS Data': False,  
            'Time Cols': []
        }
        
        # Append initial summary to before-preprocessing summary
        summary_before.append(initial_summary.copy())

        # Remove columns with more than 60% missing values
        null_percentage = df.isnull().mean() * 100
        columns_to_drop = null_percentage[null_percentage > 60].index
        df.drop(columns=columns_to_drop, inplace=True)

        # For non-time series data, if more than 4000 rows, sample 2000 rows
        if not initial_summary['TS Data'] and len(df) > 4000:
            df = df.sample(n=2000, random_state=42)
        
        # Save the preprocessed dataset
        preprocessed_path = os.path.join(preprocessed_directory, os.path.basename(file_path))
        df.to_csv(preprocessed_path, index=False)
        
        # Append summary information after preprocessing
        after_summary = initial_summary.copy()
        after_summary.update({
            'Col Cnt Af': df.shape[1],
            'Row Cnt Af': df.shape[0],
            'Miss Col Cnt Af': df.isnull().any().sum(),
        })
        summary_after.append(after_summary)

    # Convert summary information to DataFrames and sort
    summary_df_before = pd.DataFrame(summary_before).sort_values(by='Dataset name')
    summary_df_after = pd.DataFrame(summary_after).sort_values(by='Dataset name')

    # Add comparison data in the after summary table
    summary_df_after['Col Δ'] = summary_df_after['Col Cnt Af'] - summary_df_after['Col Cnt']
    summary_df_after['Row Δ'] = summary_df_after['Row Cnt Af'] - summary_df_after['Row Cnt']
    summary_df_after['Miss Col Δ'] = summary_df_after['Miss Col Cnt Af'] - summary_df_after['Missing Col Cnt']
    
    return summary_df_before, summary_df_after

# Note: Execution and example usage are commented out for development purposes.
# Adjust the following lines with actual directory paths to execute:

# before_preprocessing, after_preprocessing = preprocess_datatables('data/datatables', 'data/preprocessed_tables', columns_with_meaningful_nan)
# print("Before Preprocessing:")
# print(before_preprocessing)
# print("\nAfter Preprocessing:")
# print(after_preprocessing)
