# radiomics-ild-patterns

## 📊 binary_classification.py — Bootstrap Classification and SHAP Analysis

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

📥 Input: a CSV file: synthetic_radiomics_dataset.csv, which must include:
- Patient: Unique patient ID
- Group: Binary target class (0 or 1)
- feature_1 to feature_N: Radiomic-style feature columns

📤 Output   
All results are saved in a folder named results_YYYY-MM-DD_HH-MM-SS, including:
- CSV files with training and test performance metrics (-Train_scores.csv, -Test_scores.csv)
- Numpy file with all confusion matrices (-Confusion_matrices.npy)
- Interpolated ROC coordinates (_ROC_coords_test.csv)
- SHAP summary bar plot (-SHAP_summary.png)

▶️ How to run   
```python binary_classification.py```

🧪 Notes
- This script is designed for testing purposes and runs on fully synthetic data.
- The SHAP plot assumes class labels are [0, 1], corresponding to a binary classification scenario.

## 🧪 Synthetic Test Data

This repository includes a synthetic dataset (synthetic_radiomics_dataset.csv) created exclusively for testing and demonstration purposes. The dataset was artificially generated and does not contain any real patient data. It simulates a typical radiomics scenario, consisting of:   
- 100 observations representing individual patients
- 10 numerical features (feature_1 to feature_10) mimicking radiomic variables
- A binary group label (Group = 0 or 1) representing two hypothetical subpopulations
- A unique patient identifier (Patient)

To introduce distinguishable patterns between groups, the features for each group were sampled from normal distributions with different means (Group 0: mean = 0, Group 1: mean = 1). These differences simulate the kind of variability radiomics may capture between clinical classes, enabling testing of classification models and visualizations (e.g., SHAP analysis) in a fully anonymized and ethical context.

**This dataset is not derived from real medical imaging and should be used only for testing code and validating pipelines, not for clinical or scientific inference.**
