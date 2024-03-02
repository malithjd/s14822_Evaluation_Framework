import os
from recommendations.lida.lida_vr import save_lida_recommendations
from lida import Manager, TextGenerationConfig , llm 
import warnings
import json

# Filter out FutureWarning for seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)


dataset_dir = "data/preprocessed_tables/test"
output_dir = "recommendations/lida/charts"
gt_dir = "data/groundtruths/test"
textgen_config = TextGenerationConfig(n=1, temperature=0.3, model="gpt-3.5-turbo-0301", use_cache=True)


def count_visuals_in_json(directory):
    visual_counts = {}
    for file in os.listdir(directory):
        if file.endswith(".json"):
            file_path = os.path.join(directory, file)
            try:
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    # Count the number of keys starting with "Visual_" in each JSON file
                    visual_count = sum(key.startswith("Visual_") for key in data.keys())
                    visual_counts[file] = visual_count
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return visual_counts

# Count visuals in corresponding groundtruth tables
visual_counts = count_visuals_in_json(gt_dir)


# Call the function with the specified parameters
save_lida_recommendations(dataset_dir, output_dir, visual_counts)