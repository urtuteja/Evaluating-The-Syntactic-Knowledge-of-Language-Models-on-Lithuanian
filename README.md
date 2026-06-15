# Evaluating the Syntactic Knowledge of Language Models on Lithuanian

This repository contains the resources and evaluation results used to assess the syntactic knowledge of language models on Lithuanian through minimal sentence pair evaluation.

The study consists of **777 minimal sentence pairs**, covering **31 syntactic phenomena** and **64 error types**. Using these sentence pairs, **78 language models** were evaluated to investigate how well they capture Lithuanian syntax.

This work was developed as part of my Master's thesis:

**MSc_Logic_Thesis_Urtė_Jakubauskaitė.pdf**

---

## Background

This project extends the work introduced in:

**Jakubauskaitė, U. & Alhama, R. G. (2026)** – *Evaluating Large Language Models on Lithuanian Grammatical Cases*  
[https://aclanthology.org/2026.loreslm-1.32/](https://aclanthology.org/2026.loreslm-1.32/)

Compared to the previous work, this repository includes a considerably larger dataset and a larger number of language models.

---

## Repository Structure

```text
.
├── Lithuanian_Dataset.csv
├── Lithuanian_Dataset.xlsx
├── evaluate_pairs.py
├── MSc_Logic_Thesis_Urtė_Jakubauskaitė.pdf
└── Model Results/
    ├── Raw Results/
    │   └── 78 model output CSV files
    ├── Summaries/
    │   └── 78 model summary TXT files
    ├── error_type_difficulty_all.csv
    └── phenomenon_difficulty_all.csv
```

---

## Dataset

The repository contains two versions of the dataset:

### Lithuanian_Dataset.csv

The primary dataset used in all experiments and analyses reported in the thesis.

### Lithuanian_Dataset.xlsx

An Excel version of the dataset with color coding and formatting that provides a more convenient visual overview of the linguistic phenomena and error types.

### Important Note

The current version of the dataset still contains several minor issues that were identified during the error analysis. These limitations are discussed in detail in the thesis:

**MSc_Logic_Thesis_Urtė_Jakubauskaitė.pdf**

Future revisions of the dataset may address these issues.

---

## Evaluating Models

The script `evaluate_pairs.py` evaluates language models on the minimal sentence pairs contained in the dataset.

### Usage

```bash
python3 evaluate_pairs.py \
    --model utter-project/MODEL_NAME \
    --input Lithuanian_Dataset.csv \
    --output EVALUATED_Lithuanian_Dataset.csv \
    --token YOUR_TOKEN
```
---

## Results

The repository includes evaluation results for **78 language models**.

### Raw Results

`Model Results/Raw Results/`

Contains the original model outputs for all evaluated models in CSV format.

Each file corresponds to a single model evaluation.

### Summaries

`Model Results/Summaries/`

Contains summary reports for all 78 evaluated models.

For each model, results are aggregated by:

- Syntactic phenomenon
- Error type
- Including exceptions
- Excluding exceptions

---

## Difficulty Rankings

The repository also contains aggregate rankings of linguistic phenomena and error types based on the average performance of all evaluated models.

### phenomenon_difficulty_all.csv

Ranks syntactic phenomena from **most difficult** to **least difficult** based on average model accuracy across all 78 evaluated models.

### error_type_difficulty_all.csv

Ranks error types from **most difficult** to **least difficult** based on average model accuracy across all 78 evaluated models.

These rankings provide an overall picture of which aspects of Lithuanian syntax are the most challenging for language models.

---

## Thesis

The full methodology, dataset creation process, evaluation procedure, results, discussion (including error analysis), and limitations are described in:

**MSc_Logic_Thesis_Urtė_Jakubauskaitė.pdf**

---
