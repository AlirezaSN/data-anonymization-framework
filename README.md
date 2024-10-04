# Data Anonymization Framework

This project provides a comprehensive data anonymization framework based on Relaxed Functional Dependencies (RFDs) and optimized using Particle Swarm Optimization (PSO). The framework is demonstrated using the Adult & German Credit datasets.

## Installation Guide

1. Clone the repository:
    ```bash
    > git clone git@github.com:AlirezaSN/data-anonymization-framework.git
    > cd data_anonymization_framework
    ```

2. Create and activate a virtual environment (optional):
    ```bash
    > python3 -m venv venv
    > source venv/bin/activate
    ```

3. Install the required Python modules:
    ```bash
    > pip install -r requirements.txt
    ```

## Framework Execution Guide - Adult

1. Ensure that the Adult dataset (adult.csv) is in the same directory as the script data_anonymization_framework.py. You can download the dataset from the UCI Machine Learning Repository [here](https://archive.ics.uci.edu/dataset/2/adult).

2. Run the main script:
    ```bash
    > python data_anonymization_framework_adult.py
    ```

3. The script will load the dataset, preprocess it, extract RFDs, create generalization hierarchies, and perform data anonymization using PSO. It will print the best anonymization strategy and its fitness values (k-anonymity and information loss).

## Framework Execution Guide - German Credit

1. This dataset will be downloaded directly from the code. So, you don't have to download and embed it in the framework. If needed, you can download the dataset from the UCI Machine Learning Repository [here](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

2. Run the main script:
    ```bash
    > python data_anonymization_framework_credit.py
    ```

3. The script will load the dataset, preprocess it, extract RFDs, create generalization hierarchies, and perform data anonymization using PSO. It will print the best anonymization strategy and its fitness values (k-anonymity and information loss).

## Synthetic Dataset Generation Guide

1. Run the main script:
    ```bash
    > python synthetic_dataset_generator.py
    ```

2. The synthetic datasets will be saved as CSV files in the "synthetic_datasets" directory (or a directory of your choice). Each file will be named according to the number of QIDs it contains (e.g., `synthetic_dataset_10_QIDs.csv`).