from lida import Manager, TextGenerationConfig , llm 
from dotenv import load_dotenv
import os
import json

load_dotenv()
open_ai_key = os.getenv("OPEN_AI_KEY")

library = "matplotlib"

lida = Manager(text_gen = llm("openai", api_key=open_ai_key ))
model_type = "gpt-3.5-turbo-1106"
textgen_config_creative = TextGenerationConfig(n=1, temperature=0.5, model=model_type, use_cache=True)
textgen_config_coherenced = TextGenerationConfig(n=1, temperature=0.1, model=model_type, use_cache=True)
# persona="Data-Driven Decision Maker with Expertise in Visualization and Analytics"


def get_visual_code(filepath, count):
    summary = lida.summarize(filepath, summary_method="default")
    goals = lida.goals(summary, n=count, textgen_config=textgen_config_creative)
    codes = {}

    for goal in goals:
        # Using a unique identifier as key (e.g., "visualization_index")
        goal_id = f"{goal.visualization}_{goal.index}"
        chart = lida.visualize(summary=summary, goal=goal, library=library, textgen_config=textgen_config_coherenced)
        if chart:  # Check if the chart list is not empty
            codes[goal_id] = chart[0].code
            print(f'{goal_id}_Done!')
        else:
            print(f"{goal_id}_Failed!")
    return codes



def save_lida_recommendations(dataset_dir: str, output_dir: str, count):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(dataset_dir, filename)
            recommendations = get_visual_code(filepath, count)
            
            # Define the output path for saving recommendations
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_Lida_recs.json")
            
            # Save the recommendations in JSON format
            with open(output_path, 'w') as f:
                json.dump(recommendations, f, indent=4)
                print(f'{filename}_Lida_Recommendations Saved!')