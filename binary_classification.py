#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:32:22 2025

@author: chiaramarzi
"""

from datetime import datetime
import pandas as pd
import os
from utils import analysis

    
# Create output directory with timestamp
now = datetime.now()
formatted_date_hours = now.strftime("%Y-%m-%d_%H-%M-%S")
results_date_hours = "results_" + formatted_date_hours
results_path = "./" + results_date_hours
os.makedirs(results_path)

# Load data
df = pd.read_csv('synthetic_radiomics_dataset.csv')
M = 50  # Number of bootstrap iterations
N = df.shape[0]
analysis(N, M, df, results_path, 'Group0-Group1')