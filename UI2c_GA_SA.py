import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import warnings

# For output
fitness_list = []
annealing_list = []
gen_num = 0

# Rates for GA
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

class Successor:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome # chromosome of cities
        self.fitness = fitness # Total distance of the path (Euclidean distance)


def generate_successors(cities, population_size): # Generate random successors
    
    successors = []
    for _ in range(population_size):

        # Shuffle the cities randomly to create a random chromosome
        np.random.shuffle(cities)
        # Calculate fitness for the generated chromosome
        fitness = calculate_path_distance(cities)
        successors.append(Successor(cities, fitness))
    
    successors = sorted(successors, key=lambda successor: successor.fitness)

    return successors


def order_crossover(parents): # Order crossover
    
    parent1, parent2 = parents[0].chromosome, parents[1].chromosome

    # Create empty child chromosomes
    chromosome_length = len(parent1)
    child = [-1] * chromosome_length

    # Select a random part of the chromosome
    crossover_point1 = np.random.randint(0, chromosome_length-2)
    crossover_point2 = np.random.randint(crossover_point1, chromosome_length-1)

    # Copy the selected part from parents to children
    child[crossover_point1:crossover_point2] = parent1[crossover_point1:crossover_point2]

    # Fill the remaining positions with elements from parent2
    index = 0
    for city in parent2:
        if city not in child:
            while child[index] != -1:
                index = (index + 1) % chromosome_length
            child[index] = city

    return child


def select_parents(successors, selecting_method): # Select parents for crossover 
    
    # Use roulette wheel selection to select parents
    if selecting_method == 'r':
        
        fitness_values = np.array([individual.fitness for individual in successors])
        total_fitness = np.sum(fitness_values)
        
        selected_parents = []
        remaining_fitness_values = fitness_values.copy()  # Create a copy to track remaining fitness
        
        for _ in range(2):
            pick = np.random.uniform(0, total_fitness)
            cumulative_fitness = np.cumsum(remaining_fitness_values)
            selected_index = np.searchsorted(cumulative_fitness, pick)
            selected_parents.append(successors[selected_index])
            
            # Update remaining fitness values to exclude the selected individual
            remaining_fitness_values[selected_index] = 0
            total_fitness = np.sum(remaining_fitness_values)
                    
        return selected_parents
    
    # Use tournament selection to select parents
    elif selecting_method == 't':
        parents = []
        for _ in range(2):
            # Select random successors
            random_successors = np.random.choice(successors, 2, replace=False)
            # Select the successor with the best fitness
            best_successor = min(random_successors, key=lambda successor: successor.fitness)
            parents.append(best_successor)

        return parents


def swap_mutation(successor): # Swap mutation
    
    # Swap two random cities in the chromosome
    position1 = np.random.randint(0, len(successor))
    position2 = np.random.randint(0, len(successor))
    successor[position1], successor[position2] = successor[position2], successor[position1]
    return successor


def genetic_algorithm(population, population_size, num_generations, selecting_method):
    
    global gen_num
    same_generation_counter = 0

    # Repeat for num_generations:
    for _ in range(num_generations):

        gen_num += 1
        new_population = []
        best_individual = population[0]

        # Elitism
        new_population.append(copy.deepcopy(population[0]))

        # Create new generation
        for _ in range(population_size-1):
            
            # Select parents
            parents = select_parents(population, selecting_method)

            # Crossover
            if np.random.uniform(0, 1) < CROSSOVER_RATE:
                child = order_crossover(parents)
            
            else:
                child = copy.deepcopy(parents[0].chromosome)

            # Mutation
            if np.random.uniform(0, 1) < MUTATION_RATE:
                child = swap_mutation(child)

            # Add Successors to new population and calculate fitness
            new_population.append(copy.deepcopy(Successor(child, calculate_path_distance(child))))

        # Return the best individual in the final population
        population = sorted(new_population, key=lambda successor: successor.fitness)

        if population[0].fitness < best_individual.fitness:
            best_individual = population[0]
            same_generation_counter = 0

        if best_individual.fitness == population[0].fitness:
            same_generation_counter += 1  

            if same_generation_counter == 50:
                two_opt(population[population_size//2])
                same_generation_counter = 0
        
        # Return the best individual in the final population
        population = sorted(new_population, key=lambda successor: successor.fitness)
        
        fitness_list.append(population[0].fitness)
        print(f'\rGeneration: {gen_num}, Path length: {population[0].fitness}', end='')
        
    return best_individual


def inverse(chromosome): # Inverse part of the chromosome

    # Select a random part of the chromosome
    point1 = np.random.randint(0, len(chromosome)-1)
    point2 = np.random.randint(point1 + 1, len(chromosome))

    # Inverse the selected part of the chromosome
    chromosome[point1:point2] = chromosome[point1:point2][::-1]

    return chromosome


def simulated_annealing(cities, initial_value, decreasing_rate):

    current_solution = copy.deepcopy(Successor(cities, calculate_path_distance(cities)))
    not_improved_counter = 0
    iteration_num = 0

    while not_improved_counter < len(cities)*10*2:

        # Generate a random successor
        successor = copy.deepcopy(current_solution)

        if np.random.uniform(0, 1) < 0.5:
            successor.chromosome = swap_mutation(successor.chromosome)
        else:
            successor.chromosome = inverse(successor.chromosome)

        successor.fitness = calculate_path_distance(successor.chromosome)

        # Calculate the probability of accepting the successor
        probability = np.exp(-(successor.fitness - current_solution.fitness)/initial_value)

        # Accept the successor with probability p
        if np.random.uniform(0, 1) < probability or successor.fitness < current_solution.fitness:
            if successor.fitness < current_solution.fitness:
                not_improved_counter = 0

            current_solution = copy.deepcopy(successor)

        else:
            not_improved_counter += 1

        annealing_list.append(current_solution.fitness)

        # Decrease the initial value
        initial_value *= decreasing_rate
        iteration_num += 1

        print(f'\rIteration num: {iteration_num}, Path length: {current_solution.fitness}, Not improved: {not_improved_counter}', end='')
    
    return current_solution


def generate_cities(num_cities): # Generate random cities with x and y coordinates
    cities = []
    for _ in range(num_cities):
        x = np.random.randint(0, 200)
        y = np.random.randint(0, 200)
        city = (x, y)
        cities.append(city)

    return cities


def pythagorean_distance(x1, y1, x2, y2): # Pythagorean distance
    return math.sqrt(math.pow((x2 - x1),2) + math.pow((y2 - y1),2))


def calculate_path_distance(cities): # Fitness function  
    total_distance = 0
    
    for i in range(len(cities) - 1):
        city1 = cities[i]
        city2 = cities[i + 1]
        total_distance += pythagorean_distance(city1[0], city1[1], city2[0], city2[1])

    # Add the distance from the last city back to the first city
    total_distance += pythagorean_distance(cities[-1][0], cities[-1][1], cities[0][0], cities[0][1])

    return total_distance


# Vizualization
def plot_cities(initial, cities_ga, cities_annealing): # Plot cities
    
    initial_x, initial_y, ga_x, ga_y, sa_x, sa_y = [], [], [], [], [], []
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))  
    fig, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(15, 5))

    for city in initial:
        initial_x.append(city[0])
        initial_y.append(city[1])

    for city in cities_ga:
        ga_x.append(city[0])
        ga_y.append(city[1])
    
    for city in cities_annealing:
        sa_x.append(city[0])
        sa_y.append(city[1])
    
    # Add connection between last and first city
    initial_x.append(initial[0][0])
    initial_y.append(initial[0][1])
    ga_x.append(cities_ga[0][0])
    ga_y.append(cities_ga[0][1])
    sa_x.append(cities_annealing[0][0])
    sa_y.append(cities_annealing[0][1])

    # Figure 1
    ax1.set_title('Initial path')
    ax1.scatter(initial_x, initial_y)
    ax1.plot(initial_x, initial_y)

    ax2.set_title('Genetic algorithm')
    ax2.scatter(ga_x, ga_y)
    ax2.plot(ga_x, ga_y)

    ax3.set_title('Path length')
    ax3.plot(fitness_list)

    # Figure 2
    ax4.set_title('Initial path')
    ax4.scatter(initial_x, initial_y)
    ax4.plot(initial_x, initial_y)

    ax5.set_title('Simulated annealing')
    ax5.scatter(sa_x, sa_y)
    ax5.plot(sa_x, sa_y)

    ax6.set_title('Path length')
    ax6.plot(annealing_list)

    plt.show()


def two_opt_swap(route, i, k):
    # Create a new route by reversing a segment between indices i and k
    new_route = route[:i] + route[i:k][::-1] + route[k:]
    return new_route


def two_opt(solution):
    improved = True
    
    while improved:
        improved = False
        counter = 0
        # Iterate through the cities in the tour
        for i in range(len(solution.chromosome) - 2):
            if counter == 1:
                return
            
            for k in range(i + 2, len(solution.chromosome)):
                # Skip if adjacent
                if k - i == 1:
                    continue
                new_route = two_opt_swap(solution.chromosome, i, k)
                new_distance = calculate_path_distance(new_route)

                # Update route if new route is shorter
                if new_distance < solution.fitness:
                    solution.chromosome = new_route
                    solution.fitness = new_distance
                    improved = True
                    return
        
            if improved:
                counter += 1
                break


def main():

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    choice = input("Enter 'GA' for Genetic Algorithm, 'SA' for Simulated Annealing, or 'both' for both algorithms: ")
    
    if choice == 'GA' or choice == 'both':
        num_cities = input("Enter number of cities:")
        population_size = input("Enter population size:")
        num_generations = input("Enter number of generations:")
        selecting_method = input("Enter selecting method for genetic algorithm (r for roulette wheel selection, t for tournament selection):")
        num_cities = int(num_cities)
        population_size = int(population_size)
        num_generations = int(num_generations)
    
    if choice == 'SA' or choice == 'both':
        initial_value = input("Enter initial temp for simulated annealing:")
        decreasing_rate = input("Enter cooling rate for simulated annealing:")
        initial_value = float(initial_value)
        decreasing_rate = float(decreasing_rate)
    
    cities = generate_cities(num_cities)
    initial_city = cities

    if choice == 'GA' or choice == 'both':
        population = generate_successors(cities, population_size)
        solution1 = genetic_algorithm(population, population_size, num_generations, selecting_method)
        print(f'\nPostupnosť miest pre GA je: {solution1.chromosome}\n')
  
    if choice == 'SA' or choice == 'both':
        solution2 = simulated_annealing(cities, initial_value, decreasing_rate)
        print(f'\nPostupnosť miest pre SA je: {solution2.chromosome}\n')

    if choice == 'both':
        plot_cities(initial_city, solution1.chromosome, solution2.chromosome)
    

if __name__ == '__main__':
    main()