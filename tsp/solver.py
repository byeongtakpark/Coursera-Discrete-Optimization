import math
import random
import time
import sys
from collections import deque
from clustering import generate_initial_solution  

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def parse_input_data(input_data):
    lines = input_data.strip().split('\n')
    node_count = int(lines[0])
    points = []
    for i in range(1, node_count + 1):
        x, y = map(float, lines[i].split())
        points.append(Point(x, y))
    return points, node_count

def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def total_distance(route, points):
    return sum(length(points[route[i]], points[route[(i + 1) % len(route)]]) for i in range(len(route)))

def two_opt_swap(route, i, k):
    return route[:i] + list(reversed(route[i:k+1])) + route[k+1:]

def greedy_nearest_neighbor(points):
    node_count = len(points)
    unvisited = set(range(node_count))
    
    current = 0  
    solution = [current]  # Initialize solution with starting node
    unvisited.remove(current)

    while unvisited:
        # Select the nearest unvisited node
        nearest = min(unvisited, key=lambda node: length(points[current], points[node]))
        solution.append(nearest)
        unvisited.remove(nearest)
        current = nearest  # Move to the next node

    return solution, total_distance(solution, points)  # Return route and its total distance

def tabu_search(points, node_count, initial_solution, iterations=1000, tabu_size=100, time_limit=1800):
    """
    Parameters:
    - points: List of Point objects
    - node_count: Total number of nodes
    - initial_solution: Precomputed initial solution (random, clustering-based, or heuristic)
    - iterations: Maximum number of iterations for Tabu Search
    - tabu_size: Size of the tabu list to prevent cycling
    - time_limit: Maximum runtime in seconds

    Returns:
    - best_solution: Optimized TSP path
    - best_distance: Distance of best TSP path
    """

    current_solution = initial_solution.copy()  # Initialize with given starting solution
    tabu_list = deque(maxlen=tabu_size)  # Tabu list to store recently visited swaps
    start_time = time.time()  

    best_solution = list(current_solution)  # Store the best solution found
    best_distance = total_distance(best_solution, points)  # Compute initial distance

    for _ in range(iterations):
        if time.time() - start_time > time_limit:  
            break

        neighborhood = []  # List to store neighboring solutions
        for i in range(1, node_count - 1):
            for k in range(i + 1, node_count):
                if (i, k) not in tabu_list:  # Ensure move is not in tabu list
                    new_solution = two_opt_swap(current_solution, i, k)  
                    new_distance = total_distance(new_solution, points) 
                    neighborhood.append((new_solution, new_distance, (i, k)))

        if not neighborhood:  # If no valid neighbors found, continue to next iteration
            continue

        neighborhood.sort(key=lambda x: x[1])  # Sort solutions by distance (ascending order)
        best_neighbor, best_neighbor_distance, move = neighborhood[0]  # Choose the best neighbor

        if best_neighbor_distance < best_distance:  # Update best solution if improvement is found
            best_solution = best_neighbor
            best_distance = best_neighbor_distance

        current_solution = best_neighbor  # Move to best neighbor solution
        tabu_list.append(move)  # Add move to tabu list to prevent cycling

    return best_solution, best_distance

def solve_it(input_data):
    """
    Determines the best method for initializing the TSP solution based on problem size
    and runs Tabu Search or Greedy Local Search accordingly.

    Parameters:
    - input_data: String containing the TSP problem instance.

    Returns:
    - output_data: Formatted result with total distance and solution path.
    """
    points, node_count = parse_input_data(input_data)  # Parse input data

    # Choose initial solution strategy based on problem size
    if node_count <= 500:
        print("Using Random Initialization (No Clustering)")
        initial_solution = list(range(node_count))  # Generate a random solution
        random.shuffle(initial_solution)

    elif node_count <= 10_000:
        print("Using Clustering-Based Initialization")
        initial_solution = generate_initial_solution(points, num_clusters=10)  # Cluster-based initial solution

    else:
        print("Using Greedy Local Search Initialization (Large Problem)")
        best_solution, best_distance = greedy_nearest_neighbor(points)  # Use greedy heuristic for large problems
        return f'{best_distance:.2f} 0\n' + ' '.join(map(str, best_solution))

    # Run 2-OPT Tabu Search for problems with ≤ 10,000 nodes
    best_solution, best_distance = tabu_search(points, node_count, initial_solution)

    return f'{best_distance:.2f} 0\n' + ' '.join(map(str, best_solution))

# Main script execution: Reads input file and solves TSP
if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))  # Solve and print the result
    else:
        print('This test requires an input file. Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
