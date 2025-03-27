#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:49:25 2025

@author: chiaramarzi
"""

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of samples per group
n_per_group = 50

# Total number of features
n_features = 10

# Simulate radiomic-like features for Group 0
group0_features = np.random.normal(loc=0, scale=1, size=(n_per_group, n_features))

# Simulate radiomic-like features for Group 1 with a shifted mean to introduce some separation
group1_features = np.random.normal(loc=1, scale=1, size=(n_per_group, n_features))

# Combine into one dataset
features = np.vstack([group0_features, group1_features])

# Create feature names (e.g., 'feature_1', 'feature_2', ..., 'feature_10')
feature_names = [f'feature_{i+1}' for i in range(n_features)]

# Create patient IDs
patient_ids = [f'P{i+1:03d}' for i in range(2 * n_per_group)]

# Create group labels
groups = [0] * n_per_group + [1] * n_per_group

# Construct the DataFrame
df_synthetic = pd.DataFrame(features, columns=feature_names)
df_synthetic.insert(0, 'Patient', patient_ids)
df_synthetic.insert(1, 'Group', groups)

# Shuffle the rows (optional)
#df_synthetic = df_synthetic.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV (optional)
df_synthetic.to_csv('synthetic_radiomics_dataset.csv', index=False)

# Preview the dataset
print(df_synthetic.head())