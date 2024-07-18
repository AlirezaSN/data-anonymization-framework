# Data Anonymization Framework

This project provides a comprehensive data anonymization framework based on Relaxed Functional Dependencies (RFDs) and optimized using Particle Swarm Optimization (PSO). The framework is demonstrated using the Adult dataset.

## Installation Guide

1. Clone the repository:
    ```bash
    > git clone https://github.com/yourusername/data_anonymization_framework.git
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

## Framework Execution Guide

1. Ensure that the Adult dataset (adult.csv) is in the same directory as the script data_anonymization_framework.py. You can download the dataset from the UCI Machine Learning Repository [here](https://archive.ics.uci.edu/dataset/2/adult).

2. Run the main script:
    ```bash
    > python data_anonymization_framework.py
    ```

3. The script will load the dataset, preprocess it, extract RFDs, create generalization hierarchies, and perform data anonymization using PSO. It will print the best anonymization strategy and its fitness values (k-anonymity and information loss).

