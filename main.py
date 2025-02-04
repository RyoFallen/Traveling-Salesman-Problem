import os
import math
import random
import matplotlib.pyplot as plt



# Parse TSP file
def parse_tsp(file_path):
    start_marker = "NODE_COORD_SECTION"
    end_marker = "EOF"
    city_ids, x_coords, y_coords = [], [], []

    with open(file_path, "r") as file:
        parsing = False
        for line in file:
            if start_marker in line:
                parsing = True
                continue
            if parsing and end_marker in line:
                break
            if parsing:
                parts = line.strip().split()
                city_ids.append(int(parts[0]))
                x_coords.append(float(parts[1]))
                y_coords.append(float(parts[2]))

    return city_ids, x_coords, y_coords


# Calculate Euclidean distance
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Generate a random path
def generate_random_path(city_ids):
    path = city_ids[:]
    random.shuffle(path)
    path.append(path[0])
    return path


# Generate a greedy path starting from a random city
def generate_greedy_path(start_city, city_ids, x_coords, y_coords):
    path = [start_city]
    while len(path) < len(city_ids):
        current_city = path[-1]
        nearest_city = min(
            (city for city in city_ids if city not in path),
            key=lambda city: calculate_distance(
                x_coords[current_city - 1], y_coords[current_city - 1],
                x_coords[city - 1], y_coords[city - 1]
            ),
        )
        path.append(nearest_city)
    path.append(start_city)
    return path


# Calculate the total distance of a path
def calculate_path_distance(path, x_coords, y_coords):
    return sum(
        calculate_distance(
            x_coords[path[i] - 1], y_coords[path[i] - 1],
            x_coords[path[i + 1] - 1], y_coords[path[i + 1] - 1]
        )
        for i in range(len(path) - 1)
    )


# Initialize a population with a mix of greedy and random paths
def initialize_population(city_ids, size, greedy_ratio, x_coords, y_coords):
    population = []
    while len(population) < size:
        if random.random() < greedy_ratio:
            start_city = random.choice(city_ids)
            population.append(generate_greedy_path(start_city, city_ids, x_coords, y_coords))
        else:
            population.append(generate_random_path(city_ids))
    return population


# Tournament selection
def tournament_selection(population, x_coords, y_coords, tournament_size):
    selected = random.sample(population, tournament_size)
    return min(selected, key=lambda path: calculate_path_distance(path, x_coords, y_coords))


# Partially Mapped Crossover (PMX)
def crossover(parent1, parent2):
    size = len(parent1) - 1
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size

    # Copy segment from parent1
    child[start:end] = parent1[start:end]

    # Fill remaining positions from parent2
    for i in range(size):
        if child[i] is None:
            gene = parent2[i]
            while gene in child:
                gene = parent1[parent2.index(gene)]
            child[i] = gene

    child.append(child[0])  # Close the loop
    return child


# Mutation by reversing a random segment
def mutate(path, mutation_rate):
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(len(path) - 1), 2))
        path[start:end + 1] = reversed(path[start:end + 1])
        path[-1] = path[0]
    return path


# 2-opt optimization to improve paths locally
def two_opt(path, x_coords, y_coords):
    best_distance = calculate_path_distance(path, x_coords, y_coords)
    best_path = path[:]
    for i in range(1, len(path) - 2):
        for j in range(i + 1, len(path) - 1):
            new_path = path[:]
            new_path[i:j] = reversed(new_path[i:j])
            new_distance = calculate_path_distance(new_path, x_coords, y_coords)
            if new_distance < best_distance:
                best_distance = new_distance
                best_path = new_path
    return best_path


# Simulated Annealing for refining paths
def simulated_annealing(path, x_coords, y_coords, initial_temp, cooling_rate):
    temp = initial_temp
    current_path = path[:]
    best_path = path[:]
    best_distance = calculate_path_distance(best_path, x_coords, y_coords)

    while temp > 1:
        # Swap two cities randomly
        i, j = sorted(random.sample(range(1, len(path) - 1), 2))
        current_path[i], current_path[j] = current_path[j], current_path[i]

        # Calculate new distance
        new_distance = calculate_path_distance(current_path, x_coords, y_coords)
        if new_distance < best_distance or random.random() < math.exp((best_distance - new_distance) / temp):
            best_path = current_path[:]
            best_distance = new_distance

        # Decrease temperature
        temp *= cooling_rate

    return best_path


# Evolve the population without stagnation logic
def evolve_population_optimized(population, x_coords, y_coords, mutation_rate, tournament_size, epoch, city_ids):
    # Retain top 20% (elitism)
    elite_count = max(1, len(population) // 5)
    sorted_population = sorted(population, key=lambda path: calculate_path_distance(path, x_coords, y_coords))
    new_population = sorted_population[:elite_count]

    # Reinitialize part of the population every 25 epochs
    if epoch % 25 == 0:
        random_individuals = [generate_random_path(city_ids) for _ in range(len(population) // 5)]
        new_population.extend(random_individuals)

    # Fill the rest of the population
    while len(new_population) < len(population):
        parent1 = tournament_selection(population, x_coords, y_coords, tournament_size)
        parent2 = tournament_selection(population, x_coords, y_coords, tournament_size)
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        new_population.append(child)

    return new_population


# Plot the best path distance over epochs
def plot_progress(progress):
    plt.figure(figsize=(10, 6))
    plt.plot(progress, marker='o', linestyle='-', color='blue')
    plt.title('Best Path Distance Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.grid(alpha=0.6, linestyle='--')
    plt.show()


# Plot the best path on a 2D graph
def plot_best_path(path, x_coords, y_coords):
    plt.figure(figsize=(10, 6))
    for i in range(len(path) - 1):
        x1, y1 = x_coords[path[i] - 1], y_coords[path[i] - 1]
        x2, y2 = x_coords[path[i + 1] - 1], y_coords[path[i + 1] - 1]
        plt.plot([x1, x2], [y1, y2], 'bo-')
    plt.title('Best Path Visualization', fontsize=14)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.grid(alpha=0.6, linestyle='--')
    plt.show()


# Run the genetic algorithm without stagnation logic
def run_genetic_algorithm_with_optimizations(file_path):
    city_ids, x_coords, y_coords = parse_tsp(file_path)

    # User inputs
    population_size = int(input("Population size: "))
    mutation_rate = float(input("Mutation rate (0-1): "))
    greedy_ratio = float(input("Greedy initialization ratio (0-1): "))
    tournament_size = int(input("Tournament size: "))
    num_epochs = int(input("Number of epochs: "))

    # Initialize population
    population = initialize_population(city_ids, population_size, greedy_ratio, x_coords, y_coords)
    progress = []
    best_distance = float("inf")
    best_path = None

    # Evolve population over epochs
    for epoch in range(num_epochs):
        current_best_path = min(population, key=lambda path: calculate_path_distance(path, x_coords, y_coords))
        current_distance = calculate_path_distance(current_best_path, x_coords, y_coords)

        if current_distance < best_distance:
            best_distance = current_distance
            best_path = current_best_path

        progress.append(best_distance)

        print(f"Epoch {epoch + 1} - Best Distance: {best_distance:.2f}")
        population = evolve_population_optimized(population, x_coords, y_coords, mutation_rate, tournament_size,
                                                 epoch, city_ids)

        # Apply simulated annealing to the best path every 50 epochs
        if epoch % 50 == 0:
            best_path = simulated_annealing(best_path, x_coords, y_coords, initial_temp=1000, cooling_rate=0.95)

    # Plot results
    plot_progress(progress)
    plot_best_path(best_path, x_coords, y_coords)

    # Print the best path and distance
    print(f"Final Best Distance: {best_distance:.2f}")
    print(f"Best Path: {best_path}")


# Main execution
if __name__ == "__main__":
    tsp_file_name = input("Enter TSP file name (without extension): ")
    file_path = os.path.join(os.path.dirname(__file__), f"files/{tsp_file_name}.tsp")
    run_genetic_algorithm_with_optimizations(file_path)
