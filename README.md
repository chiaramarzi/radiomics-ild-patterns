# radiomics-ild-patterns   

## 🚀 How to Start   
To reproduce the analysis and run the code in this repository, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. **Create the environment**
   ```bash
   conda env create -f full_environment.yml
   ```
3. **Activate the environment**
   ```bash
   conda activate radiomics-ild-patterns
   ```

## 📊 [binary_classification.py](./binary_classification.py) — Bootstrap Classification and SHAP Analysis
This script performs a complete classification and interpretability pipeline using a synthetic radiomic dataset. It is designed for testing machine learning workflows based on XGBoost classifiers, including bootstrap performance evaluation and SHAP-based feature importance analysis.

🔧 Main functionalities:   
- Repeated bootstrap resampling for model training and testing (the number of bootstrap iterations can be modified by editing the M variable in the script)
- Performance evaluation through:
  - Accuracy
  - Balanced Accuracy
  - Average Precision
  - AUROC
  - Confusion matrix generation
  - ROC curve interpolation and export
- SHAP (SHapley Additive exPlanations) summary plot for model interpretability
- Automatic saving of results to a timestamped directory

📥 Input: a CSV file which must include:
- Patient: Unique patient ID
- Group: Binary target class (0 or 1)
- feature_1 to feature_N: Radiomic-style feature columns
An example of a valid CSV file is provided in [synthetic_radiomics_dataset.csv](./synthetic_radiomics_dataset.csv)

📤 Output   
All results are saved in a folder named *results_YYYY-MM-DD_HH-MM-SS*, including:
- CSV files with training and test performance metrics (suffixes: */*-Train_scores.csv*, */*-Test_scores.csv*)
- Numpy file with all confusion matrices (suffix: */*-Confusion_matrices.npy*)
- Interpolated ROC coordinates (suffix: */*_ROC_coords_test.csv*)
- SHAP summary bar plot (suffix: */*-SHAP_summary.png*)

▶️ How to run: 
```bash
python binary_classification.py
```

🛠 Notes
- This script is designed for testing purposes and runs directly on the fully synthetic data [synthetic_radiomics_dataset.csv](./synthetic_radiomics_dataset.csv)
- The SHAP plot assumes class labels are [0, 1], corresponding to a binary classification scenario.
- The script assumes the presence of *analysis* and *metrics_dispaly* functions in [utils.py](./utils.py).

## 📈 [plot.py](./plot.py) - ROC & Confusion Matrix Visualization 
This script generates graphical summaries of the model performance after bootstrap evaluation:

- **Confusion Matrix**: shows the average confusion matrix over all bootstrap test sets, including 95% bootstrap intervals for each cell. The matrix is saved as with the suffix */*-CM_scaled.png*.

- **ROC Curve**: plots the median ROC curve across bootstrap iterations, along with the 25th and 75th percentile true positive rates. The ROC plot is saved with the suffix */*_ROC_AUC_median_roc.png*.

📥 Input:
- Requires a valid *results_\** directory generated by [binary_classification.py](./binary_classification.py), containing:
  - *\*-Confusion_matrices.npy*
  - *\*_ROC_coords_test.csv*

📤 Output:
- All images are saved in a folder named *images_YYYY-MM-DD_HH-MM-SS* generated at runtime.

🛠 Notes   
- The script assumes the presence of a *make_confusion_matrix* function in [utils.py](./utils.py).

## 🧪 Synthetic Test Data
This repository includes a synthetic dataset [synthetic_radiomics_dataset.csv](./synthetic_radiomics_dataset.csv) created using the script [synthetic_dataset_creation.py](./synthetic_dataset_creation.py), exclusively for testing and demonstration purposes. The dataset was artificially generated and does not contain any real patient data. It simulates a typical radiomics scenario, consisting of:   
- 100 observations representing individual patients
- 10 numerical features (feature_1 to feature_10) mimicking radiomic variables
- A binary group label (Group = 0 or 1) representing two hypothetical subpopulations
- A unique patient identifier (Patient)

To introduce distinguishable patterns between groups, the features for each group were sampled from normal distributions with different means (Group 0: mean = 0, Group 1: mean = 1). These differences simulate the kind of variability radiomics may capture between clinical classes, enabling testing of classification models and visualizations (e.g., SHAP analysis) in a fully anonymized and ethical context.

**This dataset is not derived from real medical imaging and should be used only for testing code and validating pipelines, not for clinical or scientific inference.**
