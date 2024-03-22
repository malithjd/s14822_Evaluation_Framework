import os
from models.Lida.lida_vr import save_lida_recommendations
from models.CustomGPT_VRS.gpt4_vr import save_gpt_recommendations
from lida import Manager, TextGenerationConfig , llm 
import warnings
import json
from scripts.vl_convertor import vl_convertor


# Filter out FutureWarning for seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)


dataset_dir = "data/preprocessed_tables"
output_dir = "recommendations/lida/charts"
gt_dir = "data/vl_groundtruths"
# textgen_config = TextGenerationConfig(n=1, temperature=0.3, model="gpt-3.5-turbo-0301", use_cache=True)


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
print(visual_counts)


# save_lida_recommendations(dataset_dir="data/preprocessed_tables", output_dir = "recommendations/lida/charts3_5", model = "gpt-3.5-turbo-0125", visual_counts=visual_counts)
# save_lida_recommendations(dataset_dir="data/preprocessed_tables", output_dir = "recommendations/lida/charts4", model = "gpt-4-0125-preview", visual_counts=visual_counts)
# vl_convertor(input_dir="recommendations/lida/charts3_5", output_dir="recommendations/lida/charts3_5_vl", model = 'gpt-4-0125-preview')
# vl_convertor(input_dir="recommendations/lida/charts4", output_dir="recommendations/lida/charts4_vl", model = 'gpt-4-0125-preview')


save_gpt_recommendations(meta_dir="data/meta_summaries", output_dir="recommendations/gpt4/charts", visual_counts=visual_counts, model = "gpt-4-0125-preview")
save_gpt_recommendations(meta_dir="data/meta_summaries", output_dir="recommendations/gpt3_5/charts", visual_counts=visual_counts, model = "gpt-3.5-turbo-0125")
# vl_convertor(input_dir="recommendations/gpt4/charts", output_dir="recommendations/gpt4/charts_vl", model = 'gpt-4-0125-preview')
# vl_convertor(input_dir="recommendations/gpt3_5/charts", output_dir="recommendations/gpt3_5/charts_vl", model = 'gpt-4-0125-preview')