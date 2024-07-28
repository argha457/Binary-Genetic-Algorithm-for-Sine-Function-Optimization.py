import random
import matplotlib.pyplot as plt
import numpy as np

def initialize_population(population_size_param, chromosome_length_param):
    population = []
    for i in range(population_size_param):
        individual = [random.choice([0, 1]) for i in range(chromosome_length_param)]
        population.append(individual)
    
    return population

def decode_chromosome(chromosome):
    binary_string = ''.join(map(str, chromosome))
    decimal_value = int(binary_string, 2)
    return decimal_value

def objective_function(x):
    """
    if x == 0:
        
        return 0
    return np.sin(x) / x
    """
    
    if 10 <= x <= 25: #if we add range of X , here the range of x is (10,25)
        return np.sin(x) / x
    else:
        return 0
    

def fitness(individual):
    x = decode_chromosome(individual)
    return objective_function(x)

def select_parents(population):        # Binary Tournament Selection
    matpool = np.zeros_like(population, dtype=int)

    for i in range(len(population)):
        # Pick two random candidates between 0 and population_size - 1
        cand1 = np.random.randint(0, len(population))
        cand2 = np.random.randint(0, len(population))

        if fitness(population[cand1]) >= fitness(population[cand2]):
            selected = cand1
        else:
            selected = cand2

        matpool[i] = np.copy(population[selected])

    return matpool

def crossover(parent1, parent2, crossover_rate):     # Uniform Crossover
    if np.random.rand() < crossover_rate:
        # If crossover occurs, create a random mask
        mask = np.random.randint(0, 2, size=len(parent1))

        # Initialize child chromosomes
        child1 = np.zeros(len(parent1), dtype=int)
        child2 = np.zeros(len(parent2), dtype=int)

        for i in range(len(parent1)):
            if mask[i] == 0:  # No change if mask bit is 0
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:              # Exchange if mask bit is 1
                child1[i] = parent2[i]
                child2[i] = parent1[i]

        return child1, child2
    else:
        # If no crossover, return the parents unchanged
        return parent1, parent2

def mutate(individual, mutation_rate):
    mutated_individual = [gene if random.random() > mutation_rate else 1 - gene for gene in individual]
    return mutated_individual

def genetic_algorithm(population_size_param, chromosome_length_param, mutation_rate_param, crossover_rate_param, generations_param):
    population = initialize_population(population_size_param, chromosome_length_param)

    local_mean_fitness_values = []
    local_max_fitness_values = []
     #local_min_fitness_values = []  # Added list to store minimum fitness values
    
    #best_fitness = fitness(population[0])
    #consecutive_generations_counter = 0

    for generation in range(generations_param):
        population = sorted(population, key=fitness, reverse=True)

#         mean_fitness = sum(fitness(individual) for individual in population) / len(population)
        mean_fitness = np.mean([fitness(individual) for individual in population])
        local_mean_fitness_values.append(mean_fitness)

        local_max_fitness_values.append(fitness(population[0]))

        if fitness(population[0]) == 1:  # If the maximum possible fitness is achieved
            print(f"\nOptimal solution reached in generation {generation + 1}!")
            break
        """
        if fitness(population[0]) == best_fitness:
            consecutive_generations_counter += 1
            if consecutive_generations_counter >= consecutive_generations_threshold:
                print(f"\nConvergence reached. Stopping after {consecutive_generations_threshold} iterations.")
                break
        else:
            best_fitness = fitness(population[0])
            consecutive_generations_counter = 0
        """
        
        parents = select_parents(population)
        offspring1, offspring2 = crossover(parents[0], parents[1], crossover_rate_param)

        population[-2] = mutate(offspring1, mutation_rate_param)
        population[-1] = mutate(offspring2, mutation_rate_param)

        print(f"\nIteration {generation + 1} - Best Fitness: {local_max_fitness_values[-1]}, Mean Fitness: {local_mean_fitness_values[-1]}")
    
    # Find the best individual in the final population
    best_individual = max(population, key=fitness)
    best_x = decode_chromosome(best_individual)
    best_fitness = objective_function(best_x)
    print(f"\nThe best x value is: {best_x}")
    print(f"\nWith a fitness value of: {best_fitness}")
    print(f"\nThe Best individual: {best_individual}")

    # Assign the local variables to the global variables
    global mean_fitness_values, max_fitness_values
    mean_fitness_values = local_mean_fitness_values
    max_fitness_values = local_max_fitness_values
    print("\nFinal Population:")
    for i, individual in enumerate(population):
        print(f"Individual {i + 1}: {individual}")
    return mean_fitness_values, max_fitness_values

# Genetic Algorithm Parameters
population_size_input = int(input("Enter the population size: "))
chromosome_length_input = int(input("Enter the chromosome length: "))
generations_input = int(input("Enter the generations: "))
crossover_rate_input = float(input("Enter the crossover rate: "))
mutation_rate_input = float(input("Enter the mutation rate: "))
#consecutive_generations_threshold_input = int(input("Enter the consecutive generations threshold: "))


mean_fitness_values, max_fitness_values = genetic_algorithm(population_size_input, chromosome_length_input, mutation_rate_input, crossover_rate_input, generations_input)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(mean_fitness_values) + 1), mean_fitness_values, label='Mean Fitness')
plt.plot(range(1, len(max_fitness_values) + 1), max_fitness_values, label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.title('Genetic Algorithm - Fitness Evolution')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(range(1, len(max_fitness_values) + 1), max_fitness_values, label='Max Fitness', color='blue')
plt.scatter(range(1, len(mean_fitness_values) + 1), mean_fitness_values, label='Min Fitness', color='red')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.title('Genetic Algorithm - Max and Min Fitness Evolution')
plt.show()
