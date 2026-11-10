# Evaluating Gender and Racial Bias in Machine Learning-Based Resume Screening

## Overview
This project uses the FairCVdb synthetic resume dataset to investigate whether machine learning models reproduce demographic bias when trained on biased labels. We train Logistic Regression, Random Forest, and XGBoost models across three label conditions: Blind (unbiased), Gender-Biased, and Ethnicity-Biased.

## Setup Instructions
1. Clone this repository.
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Download the FairCVdb dataset from [FairCVtest](https://github.com/BiDAlab/FairCVtest) and place the `FairCVdb.npy` file in the `data/raw/` directory.

## How to Run the Pipeline
1. **Preprocess the Data:** Run `python src/data_preprocessor.py` to one-hot encode categorical features, standardize continuous features, and generate the processed data files in `data/processed/`.
2. **Train Baseline Models:** Run `python src/models/train.py` to train the three base models on the Blind labels and save them to `results/models/`.
3. **Run Experiments:** Execute the Jupyter notebooks in the `notebooks/` folder in order:
   - `01_exploratory_analysis.ipynb`
   - `02_train_models.ipynb`
   - `03_bias_experiment.ipynb` (Counterfactual bio test and label-type comparison)
   - `04_fairness_intervention.ipynb` (Fairness metrics and reweighting)
   - `05_results_visualization.ipynb`
   - `06_shap_analysis.ipynb`

## Key Findings
- **Accuracy hides bias:** Models trained on biased labels perform just as accurately on standard metrics as those trained on unbiased labels.
- **Gender bias survives training:** The models show strong demographic parity differences and disparate impact ratios below the 0.80 legal threshold.
- **Binarization hides ethnicity bias:** A continuous regression robustness check reveals a massive ethnicity gap (0.19) that is masked by binary classification.
- **Reweighting fails:** Sample reweighting does not fix bias that is structurally encoded in the training labels.