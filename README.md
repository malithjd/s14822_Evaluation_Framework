# Evaluation Framework For Benchmark LLM and ML-Based Visualization Recommendation Systems Using Real-World Datasets

## Description

This project automates the preprocessing of source data, extraction of ground truths, and generation of visual recommendations using various tools and scripts. It leverages Python 3.11.0 and Poetry for dependency management, providing a structured approach to handling data and generating visual insights.

## Installation

1. Clone the repository to your local machine.
2. Ensure Python 3.11.0 is installed on your system.
3. Install Poetry for dependency management. If you do not have Poetry installed, please refer to [this link](https://python-poetry.org/docs/).
4. Run `poetry install` in the project directory to create the required virtual environment and install relevant dependencies.
5. Activate the Poetry shell using `poetry shell`.

## Usage

- To preprocess the source data and extract ground truths, run:
    ```
    poetry run python main1.py
    ```
- To generate visual recommendations and save them, run:
    ```
    poetry run python main2.py
    ```

## Project Structure

```plaintext
Project Folder (root)
├── data
│   ├── datatables            <- Selected datasets (.csv) for the evaluation
│   ├── EDA_notebooks         <- Respective EDA notebooks for each datatable in datatables folder.
│   ├── preprocessed_tables   <- Preprocessed datasets (.csv) from datatables directory 
│   ├── groundtruths          <- Extracted visual codes for each notebooks 
│   ├── vl_groundtruths       <- Filtered vega-lite spec that are ready-to-go with the benchmark
├── scripts
│   ├── preprocessor.py       <- Script to preprocess the raw datasets collected
│   ├── meta_generator.py     <- Script to generate summarized metadata of the raw datasets
│   ├── visual_extractor.py <- Script to extract the visual encodings from raw ipynb notebooks
│   ├── vl_convertor.py       <- Script to convert extracted visual encodings to vega-lite spec
├── recommendations
│   ├── lida
│   │   ├── lida_vr.py        <- Script to generate and save Lida visual recommendations
│   ├── gpt4
│   │   ├── gpt4_vr.py        <- Script to generate and save GPT-4 visual recommendations with custom prompting
│   ├── multivision
│   │   ├── multivision_vr.py <- Script to generate and save Multivision visual recommendations
├── main1.py                  <- Preprocess/extract metadata summary & extracting vega-lite spec from notebooks.
├── main2.py                  <- Generate visual recommendations and save them under specified directories
├── LICENSE
├── .gitignore
├── poetry.lock
├── pyproject.toml
├── requirements.txt
```

## License

This project is licensed under the [LICENSE]().

## Contact

For any queries, please contact [malithjayasinghed@gmail.com].