import os
import json
from dotenv import load_dotenv
from scripts.benchmark_func import benchmark_visual_recommendations

load_dotenv()
open_ai_key = os.getenv("OPEN_AI_KEY")

#RUN THIS WHEN YOU ARRIVE
# benchmark_visual_recommendations(gt_dir="data/vl_groundtruths/test", recs_dir="recommendations/lida/ready_charts/test", model="gpt-3.5-turbo-0125")
benchmark_visual_recommendations(gt_dir="data/vl_groundtruths/test", recs_dir="recommendations/gpt3_5/charts_vl", output_dir="results/gpt3_5", model="gpt-3.5-turbo-0125")
benchmark_visual_recommendations(gt_dir="data/vl_groundtruths/test", recs_dir="recommendations/gpt3_5/charts_vl", output_dir="results/gpt4", model="gpt-4-0125-preview")
