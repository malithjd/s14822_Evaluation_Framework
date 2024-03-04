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

def generate_llm_messages_vl_convert(visual_encodings, filename):
    """Generate system and assistant messages for LLM based on visual encodings."""
    system_prompt = """You are an expert in python visualization grammar like seaborn, matplotlib, vega-lite, plotly, etc. 
    1. ONLY RETURNS the Vega-Lite spec for each visualization inside a VALID json.
    2. Identify the visualization grammar used within <Visuals> tags and CONVERT it to Vega-Lite spec, including title/description, encodings, mark and aggregation if applicable.
    3. Do not need non-essential attributes like $schema and data etc.
    4. In cases of ambiguity or missing information, use your best judgment to fill in gaps based on standard visualization practices.
    """
    
    assistant_message = f"Convert the following Python notebook cell's visualizations to Vega-Lite spec: <Visuals>\n{json.dumps(visual_encodings, indent=2)}</Visuals>."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": assistant_message}
    ], filename

def make_llm_call(messages, model):
    litellm.api_key = open_ai_key
    response = None  # Initialize response to None
    response_content = ""  # Initialize as empty string
    try:
        response = completion(model=model, messages=messages)
        # Extract text content from response
        if response is not None and response.choices:
            response_content = response.choices[0].message.content
            print(response_content)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Call completion_cost only if response is not None
    if response is not None:
        cost = completion_cost(completion_response=response)
        formatted_string = f"${float(cost):.10f}"
        print(formatted_string)
    else:
        formatted_string = "Error: Response is None"

    # Return the text content for file writing
    return response_content

def save_response_file(response_content, filename, output_dir):
    """Save the LLM response to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, filename)
    with open(save_path, 'w') as file:
        # Ensure that response_content is a string
        if isinstance(response_content, str):
            file.write(response_content)
        else:
            # Convert the response content to string if it's not already
            file.write(str(response_content))
    print(f"Saved response to {save_path}")


def vl_convertor(input_dir, output_dir, model):
    """Process each JSON file in the specified directory using the LLM."""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                visual_encodings = read_json_file(file_path)
                messages, filename = generate_llm_messages_vl_convert(visual_encodings, file)
                
                llm_response = make_llm_call(messages, model)
                save_response_file(llm_response, filename, output_dir)
                # Here you could process the response further if needed
