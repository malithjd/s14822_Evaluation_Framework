import json
import os
from dotenv import load_dotenv
from litellm import completion, completion_cost
import litellm

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
open_ai_key = os.getenv("OPEN_AI_KEY")
open_ai_key_4 = os.getenv("OPEN_AI_KEY_4")

def read_json_file(file_path):
    """Read and return the content of a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def read_json_file_strict(file_path):
    """Attempt to read and return the content of a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            return json.loads(file.read())
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return None

def generate_llm_messages_vl_convert(visual_encodings, filename):
    """Generate system and assistant messages for LLM based on visual encodings."""
    system_prompt = """Given a JSON formatted input file within a `<Visuals>` tag containing keys such as "Visual_1", "Visual_2", etc., each associated with a Python code snippet for generating visualizations, 
    your task is to convert these snippets into corresponding Vega-Lite specifications. Please follow these guidelines for each visualization:
    1. Analyze the Python visualization code to identify the plot type, title/description, data encodings (axes, color, etc.), and any aggregations present.
    2. Translate this analysis into a Vega-Lite specification for each visualization. Your translation should:
    - Specify the appropriate "mark" type that reflects the plot type identified in the Python code.
    - Include "encoding" details that correspond to the visualization's properties, matching axes and other characteristics found in the code, and incorporate aggregations if applicable.
    - Provide a "description" that captures the visualization's title or significant descriptive text.
    3. Retain the original "Visual_N" key for each Vega-Lite specification in the output, and exclude non-essential attributes like "$schema" and "data".
    Ensure the output JSON object retains the "Visual_N" keys, linking each Vega-Lite specification to its source visualization code snippet.

    **Expected Output Structure**
    
    {
    "Visual_1": {
        "description": "Description based on the Visual_1 code snippet, including plot type and any aggregations...",
        "mark": "bar",
        "encoding": {
        "x": {"field": "Field_X", "type": "ordinal"},
        "y": {"field": "Field_Y", "type": "quantitative", "aggregate": "sum"}
        }
    },
    "Visual_2": {
        "description": "Description based on the Visual_2 code snippet, including plot type and any aggregations...",
        "mark": "line",
        "encoding": {
        "x": {"field": "Field_X", "type": "temporal"},
        "y": {"field": "Field_Y", "type": "quantitative", "aggregate": "average"}
        }
    }
    }
    """
    
    user_message = f"Convert the following Python visualization code snippets into Vega-Lite specifications: <Visuals>\n{json.dumps(visual_encodings, indent=2)}</Visuals>."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ], filename

def make_llm_call(messages, model):
    if model=="gpt-4-0125-preview":
        litellm.api_key = open_ai_key_4
    else:
        litellm.api_key = open_ai_key

    response = None
    response_content = ""
    try:
        response = litellm.completion(model=model, messages=messages, response_format = { "type": "json_object" })
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
