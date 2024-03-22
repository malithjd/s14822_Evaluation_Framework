import os
import json
from pathlib import Path
import pandas as pd


def get_model_names(folder_name):
    evaluating_model_mapping = {
        'claude': 'claude-latest',
        'gpt4': 'gpt-4-0125-preview',
        'gpt3_5': 'gpt-3.5-turbo-0125',
    }
    
    vr_model_mapping = {
        '3_5Recs': 'gpt-3.5-turbo-0125',
        '4Recs': 'gpt-4-0125-preview',
        'LidaRecs3_5': 'Lida-gpt-3.5-turbo-0125',
        'LidaRecs4': 'Lida-gpt-4-0125-preview',
        'MVRecs': 'MultiVision',
        'ClaudeRecs': 'claude-recs-latest',
    }
    
    # Default values
    evaluating_model = 'Unknown'
    vr_model = 'Unknown'
    
    # Determine the evaluating model and vr model based on the folder name
    for key, value in evaluating_model_mapping.items():
        if key in folder_name:
            print(folder_name)
            evaluating_model = value
            break
    
    for key, value in vr_model_mapping.items():
        if key in folder_name:
            vr_model = value
            break
    
    return evaluating_model, vr_model


def add_model_values(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            evaluating_model, vr_model = get_model_names(dir)
            child_folder_path = Path(root) / dir
            for json_file in child_folder_path.glob('*.json'):
                with open(json_file, 'r') as file:
                    data = json.load(file)
                    data['vr_model'] = vr_model
                    data['evaluating_model'] = evaluating_model
                    data['dataset_id'] = json_file.stem
                with open(json_file, 'w') as file:
                    json.dump(data, file, indent=4)
            
def results_to_csv(root_dir):
    columns_order = ["dataset_id", "vr_model", "evaluating_model", "groundtruths", "recommendations", "success_count", "success_gts", "success_recs", "fail_count", "fail_gts", "fail_recs"]

    data_rows = []

    for child_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, child_folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    data_rows.append(data)

    df = pd.DataFrame(data_rows)
    df = df[columns_order]
    output_dir = "Benchmark_Results_12_Datasets.csv"
    df.to_csv(output_dir, index=False)
    print(f"Results saved in {output_dir}")