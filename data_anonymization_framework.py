import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from pyRDF import discover_rfds

# Preprocessing data
def preprocess_data(data):
    # Handle missing values, if any
    data = data.replace('?', np.nan).dropna()
    return data

# Loading the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Extracting RFDs
def extract_rfds(data):
    # Use pyRDF or similar tool to discover RFDs from the data
    rfds = discover_rfds(data)
    return rfds

# Combining RFDs to create new RFDs
def combine_rfds(rfds):
    combined_rfds = []
    for i in range(len(rfds)):
        for j in range(i + 1, len(rfds)):
            combined_rfds.append((rfds[i], rfds[j]))
    return combined_rfds

# Generate domain generalization hierarchies for categorical quasi-identifiers
def create_generalization_hierarchies():
    generalization_hierarchies = {
        'Age': [lambda x: x // 10 * 10, lambda x: 'Age Group'],
        'Workclass': [lambda x: 'Other' if x not in ['Private', 'Self-emp-not-inc'] else x],
        'Education': [lambda x: 'Higher' if x in ['Bachelors', 'Masters', 'Doctorate'] else 'Lower'],
        'Marital Status': [lambda x: 'Married' if 'Married' in x else 'Single'],
        'Occupation': [lambda x: 'White-collar' if x in ['Exec-managerial', 'Prof-specialty'] else 'Blue-collar'],
        'Race': [lambda x: 'Minority' if x != 'White' else 'White'],
        'Sex': [lambda x: 'Person'],
        'Native Country': [lambda x: 'North America' if x in ['United-States', 'Canada'] else 'Other'],
        'Relationship': [lambda x: 'In Family' if x in ['Husband', 'Wife', 'Child'] else 'Not in Family']
    }
    return generalization_hierarchies

# Anonymize data based on RFDs and generalization hierarchies
def anonymize_data(data, rfds, generalization_hierarchies, generalization_levels):
    anonymized_data = data.copy()
    for rfd in rfds:
        for attr in rfd:
            if attr in generalization_hierarchies:
                hierarchy = generalization_hierarchies[attr]
                level = generalization_levels.get(attr, 0)
                anonymized_data[attr] = anonymized_data[attr].apply(hierarchy[level])
    return anonymized_data

# Evaluate the anonymized data
def evaluate(data, anonymized_data):
    # Calculate k-anonymity
    group_sizes = anonymized_data.groupby(anonymized_data.columns.tolist()).size().reset_index(name='count')['count']
    k = group_sizes.min()
    
    # Calculate information loss
    original_cardinality = data.apply(lambda col: col.nunique()).sum()
    anonymized_cardinality = anonymized_data.apply(lambda col: col.nunique()).sum()
    info_loss = 1 - (anonymized_cardinality / original_cardinality)
    
    return k, info_loss

# Fitness function for PSO
def fitness_function(strategy, data, rfds, generalization_hierarchies):
    generalization_levels = {key: strategy[i] for i, key in enumerate(generalization_hierarchies.keys())}
    anonymized_data = anonymize_data(data, rfds, generalization_hierarchies, generalization_levels)
    k, info_loss = evaluate(data, anonymized_data)
    return k, info_loss

# Particle Swarm Optimization
def pso(data, rfds, generalization_hierarchies, num_particles=30, num_iterations=50):
    num_attributes = len(generalization_hierarchies)
    
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Particle", list, fitness=creator.FitnessMulti, speed=list, best=None)
    
    def generate_particle():
        particle = creator.Particle(random.choices(range(3), k=num_attributes))
        particle.speed = [random.uniform(-1, 1) for _ in range(num_attributes)]
        return particle
    
    def update_particle(particle, best):
        inertia = 0.5
        cognitive = 1.5
        social = 1.5
        for i in range(num_attributes):
            particle.speed[i] = (inertia * particle.speed[i] +
                                 cognitive * random.random() * (particle.best[i] - particle[i]) +
                                 social * random.random() * (best[i] - particle[i]))
            particle[i] += int(particle.speed[i])
            particle[i] = max(0, min(particle[i], 2))
    
    toolbox = base.Toolbox()
    toolbox.register("particle", generate_particle)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", update_particle)
    toolbox.register("evaluate", fitness_function, data=data, rfds=rfds, generalization_hierarchies=generalization_hierarchies)
    
    population = toolbox.population(n=num_particles)
    best = None
    
    for gen in range(num_iterations):
        for particle in population:
            particle.fitness.values = toolbox.evaluate(particle)
            if not particle.best or particle.best.fitness < particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values
            if not best or best.fitness < particle.fitness:
                best = creator.Particle(particle)
                best.fitness.values = particle.fitness.values
        
        for particle in population:
            toolbox.update(particle, best)
    
    return best, best.fitness.values

if __name__ == "__main__":
    file_path = "adult.csv"
    data = load_data(file_path)
    data = preprocess_data(data)
    rfds = extract_rfds(data)
    combined_rfds = combine_rfds(rfds)
    generalization_hierarchies = create_generalization_hierarchies()
    
    best_strategy, best_fitness = pso(data, combined_rfds, generalization_hierarchies)
    print(f"Best Strategy: {best_strategy}")
    print(f"Best Fitness (k-anonymity, Information Loss): {best_fitness}")
