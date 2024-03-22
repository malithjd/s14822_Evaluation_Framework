import os
import json
from dotenv import load_dotenv
from scripts.benchmark_func import benchmark_visual_recommendations
from utils.other_func import get_model_names, add_model_values, results_to_csv

# load_dotenv()
# open_ai_key = os.getenv("OPEN_AI_KEY")

# benchmark_visual_recommendations(gt_dir="data/vl_groundtruths", recs_dir="recommendations/lida/charts4_vl", output_dir="results/gpt4-LidaRecs4", model="gpt-4-0125-preview")
# benchmark_visual_recommendations(gt_dir="data/vl_groundtruths", recs_dir="recommendations/lida/charts3_5_vl", output_dir="results/gpt4-LidaRecs3_5", model="gpt-4-0125-preview")
# benchmark_visual_recommendations(gt_dir="data/vl_groundtruths", recs_dir="recommendations/gpt4/charts_vl", output_dir="results/gpt4-4Recs", model="gpt-4-0125-preview")
# benchmark_visual_recommendations(gt_dir="data/vl_groundtruths", recs_dir="recommendations/gpt3_5/charts_vl", output_dir="results/gpt4-3_5Recs", model="gpt-4-0125-preview")

add_model_values("results")
results_to_csv("results")