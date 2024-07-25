import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from itertools import combinations, product
import random

# Load German Credit Data dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = ["Status", "Duration", "Credit_history", "Purpose", "Credit_amount", 
                "Savings", "Employment", "Installment_rate", "Personal_status", "Debtors",
                "Residence", "Property", "Age", "Installment_plans", "Housing", 
                "Number_credits", "Job", "Liable", "Telephone", "Foreign_worker", "Credit_risk"]

df = pd.read_csv(url, delim_whitespace=True, names=column_names)

# Define quasi-identifiers and sensitive attribute
quasi_identifiers = ["Age", "Job", "Housing", "Credit_amount", "Duration"]
sensitive_attribute = "Credit_risk"

# Convert categorical data to numeric codes
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Binning for numerical quasi-identifiers
bins = {"Age": 10, "Credit_amount": 10, "Duration": 10}
for col, n_bins in bins.items():
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    df[col] = est.fit_transform(df[[col]])

print("Preprocessed DataFrame:\n", df.head())

# Function to discover RFDs in the dataset
def discover_rfds(df, quasi_identifiers):
    rfds = []
    for i, qid1 in enumerate(quasi_identifiers):
        for j, qid2 in enumerate(quasi_identifiers):
            if i != j:
                rfds.append({qid1: qid2})
    return rfds

rfds = discover_rfds(df, quasi_identifiers)
print("Discovered RFDs:\n", rfds)

# Function to combine RFDs
def combine_rfds(rfds):
    combined_rfds = []
    for i, rfd1 in enumerate(rfds):
        for j, rfd2 in enumerate(rfds):
            if i != j:
                combined_rfd = {**rfd1, **rfd2}
                combined_rfds.append(combined_rfd)
    return combined_rfds

combined_rfds = combine_rfds(rfds)
print("Combined RFDs:\n", combined_rfds)

# Define generalization hierarchies for categorical attributes
generalization_hierarchies = {
    "Job": ["unemployed", "unskilled", "skilled", "management"],
    "Housing": ["rent", "own", "free"]
}

# Generate all possible generalization strategies
def generate_strategies(quasi_identifiers, generalization_hierarchies):
    strategies = []
    generalization_levels = {qid: range(len(generalization_hierarchies[qid])) if qid in generalization_hierarchies else [0] for qid in quasi_identifiers}
    for levels in product(*generalization_levels.values()):
        strategy = dict(zip(quasi_identifiers, levels))
        strategies.append(strategy)
    return strategies

strategies = generate_strategies(quasi_identifiers, generalization_hierarchies)
print("Generated Strategies:\n", strategies)

# Function to calculate k-anonymity and information loss for a strategy
def evaluate_strategy(strategy, df, quasi_identifiers, sensitive_attribute):
    # Apply generalization based on strategy
    df_generalized = df.copy()
    for qid, level in strategy.items():
        if qid in generalization_hierarchies:
            hierarchy = generalization_hierarchies[qid]
            max_level = len(hierarchy)
            df_generalized[qid] = df[qid].apply(lambda x: min(level, max_level - 1))

    # Calculate k-anonymity
    k_anonymity = df_generalized.groupby(quasi_identifiers).size().min()

    # Calculate information loss (example: normalized count of generalized values)
    info_loss = df_generalized[quasi_identifiers].nunique().mean() / df[quasi_identifiers].nunique().mean()

    return k_anonymity, info_loss

# PSO to find the best strategies
def pso_optimize(strategies, df, quasi_identifiers, sensitive_attribute, num_particles=30, iterations=100):
    particles = random.sample(strategies, num_particles)
    global_best_strategy = None
    global_best_fitness = float('inf')
    
    for _ in range(iterations):
        for particle in particles:
            k_anonymity, info_loss = evaluate_strategy(particle, df, quasi_identifiers, sensitive_attribute)
            fitness = info_loss - k_anonymity  # Hypothetical fitness function
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_strategy = particle
                
    return global_best_strategy, global_best_fitness

best_strategy, best_fitness = pso_optimize(strategies, df, quasi_identifiers, sensitive_attribute)
print("Best Strategy:\n", best_strategy)
print("Best Fitness:\n", best_fitness)
