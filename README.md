<img src="./assets/Lole_Logo.jpg" width=60%>
<br>

## INTRODUCTION - Lole Networks
The project explores the generation and processing of responses from Large Language Models (LLMs) under normal and hallucinatory conditions. It includes scripts to generate responses, process them using various logic gates (OR, AND, NOT XOR, NOT AND), calculate semantic similarity scores, and create plots. This README provides a guide on the structure, usage, and requirements to replicate the experiments or extend the work.

## Directory Structure
The repository is organized into the following directories and files:

goodresponse.py: Script to generate responses from LLMs under normal conditions.
hallucinatingresponse.py: Script to generate responses from LLMs simulating hallucination conditions.
responseprocessing.py: Script for processing LLM responses, including creating word clouds, calculating semantic similarity, and identifying common words.
integrated_logic.py: The main script that runs the experiments, combining the functionalities of the above scripts, and writes results to a CSV file.
/plotting: Contains scripts for generating plots based on the logic scores and responses generated by the LLMs. Example Logic scores attached.
/Checkpoints: Includes model configurations, tokenizers, and vocab files necessary for running the LLMs.
Logic_scores.csv: CSV file where the similarity scores and other metrics are recorded.
requirements.txt: Lists all the necessary Python packages to run the scripts.

## Setup and Requirements
To run the scripts and replicate the experiments, follow these steps:

* Environment Setup: Ensure you have Python 3.9+ installed. It's recommended to use a virtual environment.
* Install Dependencies: Install all required packages using the provided requirements.txt file.

```bash
# install key dependencies
pip install -r requirements.txt
```

* API Keys and Model Directories: The scripts require API keys for Hugging Face and OpenAI, as well as the path to the Cerebras model directory. Ensure you have these keys and update the scripts with your own keys and paths.

* Plotting: 
First step would be uploading generated Logic_scores. csv file in the plotting folder. The following example scripts should work post that point, if the respective row numbers are provided:

```bash
# Example
row_ranges = {'T(0.2,0.2)': (189, 200), 'T(0.2,0.7)': (202, 214), 'T(0.2,1.3)': (216, 228), 'T(0.2,2.0)': (230, 241) }
```

Example1: Plotting Generic trends, Intrinsic hallucinations and Extrinsic Hallucinations is available with this code
```bash
# Generic trends
python plotting_generic_trends.py
```
Example2: Plotting the evolution of hallucination with changing temperature is available in this code
```bash
# Temperature based evolution
python plot_temp_average_score.py
```
Example3: Plotting average scores of the logic gates
```bash
# main code
python plotting_average_scores.py
```

## Generating and Processing Responses

The core functionality is encapsulated in three classes: GoodResponse, HallucinatedResponse, and ResponseProcessor. These handle the generation of LLM responses under different conditions and their subsequent processing.

To generate responses: Modify the prompts in integrated_logic.py and run the script.
To process responses: The ResponseProcessor class includes methods for logic gate pooling, creating word clouds, and calculating semantic similarities.

## Extending the Work
To add new models or response processing techniques, extend the existing classes or add new scripts in the respective folders.
For plotting and analysis, add or modify scripts in the /plotting directory.

## Contributing
Contributions are welcome! If you have improvements or new features, please submit a pull request or open an issue.

## Citation
If you use this work in your research, please cite the accompanying paper (details provided in the submission).

## License
MIT License

