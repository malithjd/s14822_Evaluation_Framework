import os
import json
from dotenv import load_dotenv
from scripts.vl_convertor import read_json_file_strict, make_llm_call, save_response_file


load_dotenv()
open_ai_key = os.getenv("OPEN_AI_KEY")

def generate_llm_messages_benchmark(gt_content, recs_content, base_name):
    
    system_prompt = """
    As an AI, your function is to act as an evaluator for code snippet recommendations concerning visual representations. 
    These are enclosed within specific tags: recommendations in <RECS>...</RECS> and ground truth visualizations in <GROUND>...</GROUND>. 
    Your primary objective is to compare these two sets, focusing on chart types, axis attributes (including orientation and data fields/aggregations), 
    and titles/descriptions to assess their alignment.

    **Evaluation Criteria:**
    1. Chart Type Identity: The recommended and ground truth visuals must share the same chart type.
    2. Axis Alignment: Ensure axes are identical, considering their orientation and any specified data fields or aggregations.
    3. Title/Description Relevance: Use these elements to further confirm a match, though they are secondary to the visual structure.

    **Output Requirements:**
    Compile your findings into a JSON format with:
    - "groundtruths": Total count of unique visuals within <GROUND> tags.
    - "recommendations": Total count of unique recommendations within <RECS> tags.
    - "success_count": Number of matches between recommendations and ground truths.
    - "success_gts" & "success_recs": Keys of matched visuals from ground truths and recommendations, respectively.
    - "fail_count": Number of visuals in ground truths not matched with any recommendations.
    - "fail_gts" & "fail_recs": Keys of unmatched visuals from ground truths and recommendations, respectively.

    Your analysis should rigorously apply the criteria above to determine the efficacy of the visual recommendations compared to the ground truths. 

    **Expected JSON Output Example[Stick to this always]:**
    
    {
    "groundtruths": 5,
    "recommendations": 5,
    "success_count": 3,
    "success_gts": ["Key of GT 1", "Key of GT 2", "Key of GT 3"],
    "success_recs": ["Key of Rec 1", "Key of Rec 2", "Key of Rec 3"],
    "fail_count": 2,
    "fail_gts": ["Key of GT 4", "Key of GT 5"],
    "fail_recs": ["Key of Rec 4", "Key of Rec 5"]
    }
    """

    user_prompt = f"""
    I'm working with visual recommendations and ground truth visualizations. 
    Your task is to assess the recommendations in relation to the ground truths, focusing on key aspects such as chart type, axes, data fields, and titles to identify matches. 
    The code snippets for these visuals are provided below:

    **Recommendations:**
    <RECS>
    {recs_content}
    </RECS>

    **Ground Truths:**
    <GROUND>
    {gt_content}
    </GROUND>
    """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ], base_name

def benchmark_visual_recommendations(gt_dir, recs_dir, output_dir, model):
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]
    recs_files = [f for f in os.listdir(recs_dir) if "_recs.json" in f]
    
    for gt_file in gt_files:
        base_name = gt_file.replace('.json', '')
        matched_files = [f for f in recs_files if f.startswith(base_name)]
        
        for recs_file in matched_files:
            gt_path = os.path.join(gt_dir, gt_file)
            recs_path = os.path.join(recs_dir, recs_file)
            
            gt_content = read_json_file_strict(gt_path)
            recs_content = read_json_file_strict(recs_path)
            
            if gt_content is not None and recs_content is not None:
                messages, filename = generate_llm_messages_benchmark(gt_content, recs_content, base_name)
                response_content = make_llm_call(messages, model)
                print(f"Benchmark process completed for {filename}")
                save_response_file(response_content=response_content, filename=f'{base_name}.json', output_dir=output_dir)
            else:
                print(f"Skipping due to JSON errors: {gt_file}")
        if not matched_files:
            print(f"No matching recommendations found for {gt_file}")
    