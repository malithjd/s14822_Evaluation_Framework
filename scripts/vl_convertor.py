import json
import os
from dotenv import load_dotenv
from litellm import completion, completion_cost
import litellm

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
open_ai_key = os.getenv("OPEN_AI_KEY")

def read_json_file(file_path):
    """Read and return the content of a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def generate_llm_messages(visual_encodings, filename):
    """Generate system and assistant messages for LLM based on visual encodings."""
    system_prompt = """You are an expert in python visualization grammar like seaborn, matplotlib, vega-lite, plotly, etc. 
    1. ONLY RETURNS the Vega-Lite spec for each visualization inside a valid json.
    2. Identify the visualization grammar used within <Visuals> tags and CONVERT it to Vega-Lite spec, including title/description, encodings, mark and aggregation if applicable.
    3. Do not need non-essential attributes like $schema and data etc.
    4. In cases of ambiguity or missing information, use your best judgment to fill in gaps based on standard visualization practices.
    """
    
    assistant_message = f"Convert the following Python notebook cell's visualizations to Vega-Lite spec: <Visuals>\n{json.dumps(visual_encodings, indent=2)}</Visuals>."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": assistant_message}
    ], filename

def make_llm_call(messages, filename, model='gpt-3.5-turbo'):
    litellm.api_key = open_ai_key
    response = None
    try:
        response = litellm.completion(messages=messages, model=model)
        response_content = response.choices[0].message.content
        print(response_content)
    except Exception as e:
        print(f"An error occurred: {e}")
        response_content = ""
    cost = completion_cost(completion_response=response)
    formatted_string = f"${float(cost):.10f}"
    print(formatted_string)

    # Save the response content to a file in 'data/vl_groundtruths'
    save_response_file(response_content, filename)
    return response

def save_response_file(response_content, filename):
    """Save the LLM response to a file."""
    directory = "data/vl_groundtruths"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, filename)
    with open(save_path, 'w') as file:
        file.write(response_content)
    print(f"Saved response to {save_path}")

def vl_convertor(directory):
    """Process each JSON file in the specified directory using the LLM."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                visual_encodings = read_json_file(file_path)
                messages, filename = generate_llm_messages(visual_encodings, file)
                
                llm_response = make_llm_call(messages, filename)
                # Here you could process the response further if needed
