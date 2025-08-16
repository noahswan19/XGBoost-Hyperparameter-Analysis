# XGBoost Hyperparameter Analysis

A comprehensive XGBoost hyperparameter optimization project for tennis match prediction using ATP match data from 2003-2023.

## Overview
This project implements automated hyperparameter tuning for XGBoost models using the Hyperopt library, with multiple search strategies including full parameter space exploration and percentile-based narrowing. Details of the project can be found at the Medium article here:

## Features
- Automated data loading from Jeff Sackmann's tennis database
- Multiple hyperparameter search strategies
- Model performance comparison and visualization
- Comprehensive logging and model persistence

## Installation
```bash
git clone <your-repo-url>
cd XGBoost-Hyperparameter-Analysis
pip install -r requirements.txt
```

## Usage
```bash
# Run full hyperparameter optimization
python src/main.py

# Generate performance charts
python src/charting_execution.py
```

## Outputs
- Trained XGBoost models saved in `models/` directory
- Performance charts saved in `charts/` directory
- Detailed log in `logs/` directory

## Project Structure
```
src/
├── main.py                 # Main hyperparameter optimization
├── data_preparation.py     # Data processing pipeline functions
├── xgboost_modeling.py     # XGBoost training and tuning functions
├── charting_execution.py   # Performance visualization
└── charting_functions.py   # Charting utilities
```


This project is licensed under the MIT License – see the LICENSE file for details.
