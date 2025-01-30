# Genetic Algorithm for Traveling Salesman Problem (TSP)

## Overview
This project implements a **Genetic Algorithm (GA)** to solve the **Traveling Salesman Problem (TSP)**. The algorithm optimizes a set of routes by simulating natural selection principles, including mutation, crossover, and selection.

## Features
- **Multiple Selection Methods:** Tournament selection for evolutionary pressure.
- **Crossover Techniques:** Partially Mapped Crossover (PMX) to maintain genetic diversity.
- **Mutation Strategies:** Inversion mutation for local optimization.
- **Local Search Optimization:** Uses **2-opt** and **Simulated Annealing** to improve final solutions.
- **Hybrid Population Initialization:** Mix of **random** and **greedy** approaches.
- **Elitism & Reinitialization:** Preserves top-performing solutions and periodically replaces weak ones.

## Tested Parameters
The algorithm has been tested with different parameters to determine the most efficient configurations.

### **Parameter Variations**
- **Population Size:** 200, 1000
- **Mutation Probability:** 0.3
- **Greedy Initialization Ratio:** 0.5, 0.8
- **Tournament Size:** 10, 50
- **Number of Epochs:** 100, 1000



