#! G:\Internship\Surge\For Evaluation\Project\.venv\Scripts\python.exe

import nbformat
import os
import json

def is_visualization_code(cell):
    visualization_keywords = ['plt.', 'sns.', 'plotly.', 'fig', 'ax.', 'go.', 'bk.']
    setup_keywords = ['import ', 'from ', 'sns.set', '%matplotlib', 'warnings.filterwarnings', 'plt.style.use']

    if cell['cell_type'] != 'code':
        return False

    actionable_lines = []  # Lines that potentially contain visualization commands

    for line in cell['source'].splitlines():
        # Skip comments and empty lines
        if line.strip().startswith('#') or not line.strip():
            continue
        # If it's a setup line but not the sole content of the cell, continue to actionable lines
        if any(setup_keyword in line for setup_keyword in setup_keywords):
            continue
        actionable_lines.append(line)

    # Check actionable lines for visualization commands
    return any(keyword in line for line in actionable_lines for keyword in visualization_keywords)



def extract_visualization_codes(notebook_path: str) -> list:
    with open(notebook_path, 'r', encoding='utf-8') as nb_file:
        nb_contents = nbformat.read(nb_file, as_version=4)
    
    return [cell['source'] for cell in nb_contents['cells'] if is_visualization_code(cell)]

def save_visualization_codes_to_json(visual_codes: list, output_path: str):
    visualizations_json = {f"Visual_{i+1}": code for i, code in enumerate(visual_codes)}
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(visualizations_json, json_file, indent=4)

def process_notebook(notebook_file: str, notebooks_dir: str, output_dir: str):
    notebook_path = os.path.join(notebooks_dir, notebook_file)
    visual_codes = extract_visualization_codes(notebook_path)
    
    output_filename = os.path.splitext(notebook_file)[0] + "_visuals.json"
    output_path = os.path.join(output_dir, output_filename)
    
    save_visualization_codes_to_json(visual_codes, output_path)
    
    return len(visual_codes)

def main():
    notebooks_dir = "./data/EDA_notebooks"
    output_dir = "./data/groundtruths"
    os.makedirs(output_dir, exist_ok=True)
    
    for notebook_file in os.listdir(notebooks_dir):
        if notebook_file.endswith('.ipynb'):
            num_visuals = process_notebook(notebook_file, notebooks_dir, output_dir)
            print(f"Processed {notebook_file}: {num_visuals} visualization(s) identified and saved.")

if __name__ == "__main__":
    main()
