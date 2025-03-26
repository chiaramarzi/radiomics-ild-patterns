# radiomics-ild-patterns

## 🧪 Synthetic Test Data

This repository includes a synthetic dataset (synthetic_radiomics_dataset.csv) created exclusively for testing and demonstration purposes. The dataset was artificially generated and does not contain any real patient data. It simulates a typical radiomics scenario, consisting of:   
- 100 observations representing individual patients
- 10 numerical features (feature_1 to feature_10) mimicking radiomic variables
- A binary group label (Group = 0 or 1) representing two hypothetical subpopulations
- A unique patient identifier (Patient)

To introduce distinguishable patterns between groups, the features for each group were sampled from normal distributions with different means (Group 0: mean = 0, Group 1: mean = 1). These differences simulate the kind of variability radiomics may capture between clinical classes, enabling testing of classification models and visualizations (e.g., SHAP analysis) in a fully anonymized and ethical context.

**This dataset is not derived from real medical imaging and should be used only for testing code and validating pipelines, not for clinical or scientific inference.**
