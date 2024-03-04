import os
import json
from dotenv import load_dotenv
import litellm
from scripts.vl_convertor import read_json_file, save_response_file, make_llm_call


load_dotenv()
open_ai_key = os.getenv("OPEN_AI_KEY")

def generate_llm_messages_gpt4(meta_summary, filename, count):
    
    system_prompt = """
    Your task as an AI specialized in data visualization is to analyze dataset summaries provided within <SUM> and </SUM> tags and recommend the most suitable visualizations techniques. Your recommendations should be limited to scatterplots, bar charts, line charts, pie charts, and area charts. You must provide N number of recommendations which will be specified within <N></N> tags. Utilize matplotlib to structure your recommendations, focusing on specifying relevant column names and any necessary aggregation methods.

    For each recommended visualization type, provide a Python code snippet. These snippets should be structured to easily adapt to the dataset described in the summary. Your response should consist solely of these code snippets, formatted as a JSON object. Use keys "Rec_Gpt_1", "Rec_Gpt_2", etc., to indicate the order of recommendations, with each key mapping to a Python code snippet that illustrates how to implement the suggested visualization type. Focus on the logic and structure of the visualization code rather than including inline dataset values.

    Ensure the JSON object is clearly formatted, with each recommendation easily identifiable and actionable. The recommendations should leverage matplotlib effectively, demonstrating an understanding of data visualization principles suitable for the dataset's characteristics as summarized.

    Example structure for the expected JSON output (note: specifics will vary based on dataset summary):
    {
    "Rec_Gpt_1": "plt.figure(figsize=(10, 6))\\nax = df.groupby('column_name')['other_column'].agg_method().plot(kind='bar', rot=0)\\nplt.title('Your Title Here')\\nplt.xlabel('X-axis Label')\\nplt.ylabel('Y-axis Label')",
    "Rec_Gpt_2": "plt.figure(figsize=(10, 6))\\nsns.lineplot(data=df, x='column_name', y='other_column', estimator='agg_method', ci=None)\\nplt.title('Your Title Here')\\nplt.xlabel('X-axis Label')\\nplt.ylabel('Y-axis Label')"
    }

    Your goal is to provide clear, concise, and structured visualization recommendations that directly respond to the dataset summary, enabling users to create effective visualizations with matplotlib."""

    user_prompt = f"Based on the following meta data summary of the dataset, can you recommend the most suitable types of visualizations?: <SUM>\n{json.dumps(meta_summary, indent=2)}</SUM>. I need <N>{count}</N> recommendations"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": user_prompt}
    ], filename


def save_gpt_recommendations(meta_dir: str, output_dir: str, visual_counts):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(meta_dir):
        if filename.endswith(".json"):
            base_filename = filename.replace('_summary.json', '')
            json_filename = f"{base_filename}.json"
            count = visual_counts.get(json_filename)

            if count is not None:
                file_path = os.path.join(meta_dir, filename)
                meta_summary = read_json_file(file_path)
                messages, filename = generate_llm_messages_gpt4(meta_summary, filename, count)
                llm_response = make_llm_call(messages, model="gpt-3.5-turbo-1106")
                save_response_file(response_content=llm_response, filename=f'{base_filename}_gpt_recs.json', output_dir=output_dir)
            else:
                print(f"No visual count found for {filename}, skipping...")

