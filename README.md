# Using VLMs for understanding system diagrams

This project explores how well different Vision-Language Models (VLMs) can identify components and connections in technical system diagrams.

## Experiments

Three models were tested on six system diagrams:

- Qwen3.8-27B-FP8
- Kimi-K2.6
- GLM-5.3-Flash

The project compares seven different methods, which are Zero-shot, One-shot, Context, Notation, OCR, Two-stage, and Context + Notation. Two adapted methods from previous research were also tested. Each experiment was repeated five times, and a majority voting result was calculated for each combination. The results were evaluated manually against a manually created Ground Truth, and precision, recall and F1-score were calculated based on the evaluations in `evaluation/manual_results_csv`.

## Project structure

- `main.py` - Used for running the experiments.
- `config.py` - Is used to select the relevant model, diagram and method(s).
- `methods/` - Contains OCR, One-shot and majority voting.
- `from papers/` - Contains the adapted methods from previous research.
- `prompts/` - Contains the prompts that were used in this experiment.
- `resources/` - Contains the additional context and diagram notation.
- `input/` - Contains the relevant system diagrams.
- `ground_truth/` - Contains the manually created Ground Truth.
- `evaluation/` - Contains the evaluated results.
- `results/` - Contains the saved results from the model, as well as which configurations were used.

## How to run

First install the required packages:

```bash
pip install -r requirements.txt
```

Then create a `.env` file with your NTNU API key:

```env
NTNU_API_KEY=your_api_key
```

Select a model, diagram and method that you want to test in `config.py`, then run:

```bash
python main.py
```

Keep in mind that OCR also requires Tesseract to be installed separately.

The experiments use NTNU's IDUN API, so access is required to run them with the original setup. However, you can also use other models from other providers. In order to do this:

1. Change `base_url` in `main.py` to the API endpoint of your provider.
2. Update `api_key` in `main.py` and add your API key to the `.env` file.
3. Change `MODEL` in `config.py` to the model you want to use.

Just make sure that the selected model supports image inputs.

## Results

The models generally identified components more accurately than connections. GLM achieved the highest connection F1-score of 97.2% by using Context + Notation and majority voting. The performance varied between models and methods, and this shows that additional instructions do not always improve the results.

## Note about the results folder

The `results/` folder contains many files from both the final experiments and earlier tests. Therefore, it might be difficult to find the exact outputs that were used in the report. However, the files are included as evidence for anyone who wants to verify the results.
