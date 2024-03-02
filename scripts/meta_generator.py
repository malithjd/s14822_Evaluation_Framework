#! G:\Internship\Surge\For Evaluation\Project\.venv\Scripts\python.exe

import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any
from scipy.stats import pearsonr
from pathlib import Path

# Revised version of MetaGenerator without LLM integration
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


class MetaGenerator:
    def __init__(self, dataframe: pd.DataFrame, filepath: str):
        self.df = dataframe
        self.dataset_name = os.path.basename(filepath).split('.')[0]

    def infer_semantic_types(self) -> Dict[str, str]:
        semantic_types = {}
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                if self.df[column].str.contains('http').any():
                    semantic_types[column] = 'URL'
                elif self.df[column].str.contains('@').any():
                    semantic_types[column] = 'Email'
                else:
                    semantic_types[column] = 'Text'
            elif np.issubdtype(self.df[column].dtype, np.number):
                semantic_types[column] = 'Numeric'
            elif np.issubdtype(self.df[column].dtype, np.datetime64):
                semantic_types[column] = 'DateTime'
            else:
                semantic_types[column] = 'Unknown'
        return semantic_types

    def data_quality_insights(self) -> Dict[str, Any]:
        quality_report = {
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicate_rows': self.df.duplicated().sum(),
            'columns_stats(For_categorical_variables_TopK)': {}
        }
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                top_values = self.df[column].value_counts().head(5).to_dict()
            else:
                top_values = {
                    'mean': round(self.df[column].mean(), 4),
                    'std': round(self.df[column].std(), 4),
                    'min': round(self.df[column].min(), 4) if pd.api.types.is_float_dtype(self.df[column]) else self.df[column].min(),
                    'max': round(self.df[column].max(), 4) if pd.api.types.is_float_dtype(self.df[column]) else self.df[column].max()
                }
            quality_report['columns_stats(For_categorical_variables_TopK)'][column] = top_values
        return quality_report

    
    def sample_example_values(self) -> Dict[str, List[Any]]:
        """
        Sample up to 5 unique values from each column to provide as examples.
        """
        example_values = {}
        for column in self.df.columns:
            unique_values = pd.Series(self.df[column].unique())
            sampled_values = unique_values.sample(n=min(2, len(unique_values)), replace=False).tolist()
            example_values[column] = sampled_values
        return example_values

    def correlation_analysis(self) -> Dict[str, float]:
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        correlations = {}
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                # Drop rows with NaN in either column to avoid ValueError in correlation calculation
                valid_data = self.df[[col1, col2]].dropna()
                if not valid_data.empty:
                    corr, _ = pearsonr(valid_data[col1], valid_data[col2])
                    correlations[f'{col1}-{col2}'] = round(corr, 4)
        return correlations

    def generate_metadata_summary(self) -> dict:
        summary = {
            "dataset_name": self.dataset_name,
            "semantic_types": self.infer_semantic_types(),
            "example_values": self.sample_example_values(),
            "data_quality_insights": self.data_quality_insights(),
            "correlation_analysis": self.correlation_analysis(),
        }
        return summary


def generate_and_save_summaries(input_dir, output_dir):
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Iterate over all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path)
            
            # Generate metadata summary
            meta_gen = MetaGenerator(df, file_path)
            metadata = meta_gen.generate_metadata_summary()
            
            # Serialize and save the summary
            output_path = os.path.join(output_dir, f"{Path(filename).stem}_summary.json")
            with open(output_path, "w") as json_file:
                json.dump(metadata, json_file, indent=4, cls=NumpyEncoder)

# # Specify the input and output directories
# input_dir = "data/preprocessed_tables"
# output_dir = "data/meta_summaries"

# # Call the function to generate and save summaries
# generate_and_save_summaries(input_dir, output_dir)