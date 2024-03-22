from scripts.preprocessor import preprocess_datatables
from scripts.meta_generator import generate_and_save_summaries
from scripts.visual_extractor import process_notebook
from scripts.vl_convertor import vl_convertor
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)

#---- Preprocessing the Raw Datasets (.csv) from kaggle ------
### If there is any NaN value that is very important and needed to have in the dataset as it is.. Please include the exact column names of them here.
### It will be ignored from the basic data preprocceing methods
### For more accurate results, Please review the raw/original dataset and make neccesery edits there before running main1.py
columns_with_meaningful_nan = {}
before_preprocessing, after_preprocessing = preprocess_datatables('data/datatables', 'data/preprocessed_tables', columns_with_meaningful_nan)

print(after_preprocessing)
print("Proprocessing Done!")


#---- Getting the summary metadata from the preprocessed datasets -----
input_dir = "data/preprocessed_tables"
output_dir = "data/meta_summaries"

generate_and_save_summaries(input_dir, output_dir)
print("Meta Summary Generated and Saved!")


#---- Extraction of visual encodings in EDA notebooks saving them in data/groundtruth -----
notebooks_dir = "data/EDA_notebooks"
output_dir = "data/groundtruths"
os.makedirs(output_dir, exist_ok=True)

for notebook_file in os.listdir(notebooks_dir):
        if notebook_file.endswith('.ipynb'):
            num_visuals = process_notebook(notebook_file, notebooks_dir, output_dir)
            print(f"Processed {notebook_file}: {num_visuals} visualization(s) identified and saved.")

print("Extraction of Visual Encodings Done!")


#---- Visual Encodings to Vega-Lite Spec Convertion -----
vl_convertor("data/groundtruths", "data/vl_groundtruths", model='gpt-4-0125-preview')
print("vl-conversion Done!")


