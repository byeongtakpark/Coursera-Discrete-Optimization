import math
import random
import time
import sys
import gurobipy as gp

from collections import deque
# from clustering import generate_initial_solution  
from gurobipy import GRB

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
    # Initialize solution with starting node

    solution = [current]  
    unvisited.remove(current)

    while unvisited:
        # Select the nearest unvisited node
        nearest = min(unvisited, key=lambda node: length(points[current], points[node]))
        solution.append(nearest)
        unvisited.remove(nearest)
        current = nearest  # Move to the next node

    return solution, total_distance(solution, points)  

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

def gurobi_solver(points, node_count, time_limit=600):
    # Create a distance matrix
    distance = {}
    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                distance[i, j] = length(points[i], points[j])  # Compute Euclidean distance
            else:
                distance[i, j] = 0  # Distance from a node to itself is zero

    # Create a new Gurobi model
    m = gp.Model("TSP")
    m.Params.OutputFlag = 0  # Suppress solver output

    # Decision variables: x[i, j] = 1 if the tour includes edge (i, j), 0 otherwise
    x = {}
    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    m.update()

    # Objective function: Minimize the total travel distance
    m.setObjective(
        gp.quicksum(distance[i, j] * x[i, j] 
                    for i in range(node_count) 
                    for j in range(node_count) if i != j),
        GRB.MINIMIZE
    )

    # Constraint 1: Each node must have exactly one outgoing edge
    for i in range(node_count):
        m.addConstr(
            gp.quicksum(x[i, j] for j in range(node_count) if i != j) == 1,
            name=f"out_{i}"
        )

    # Constraint 2: Each node must have exactly one incoming edge
    for j in range(node_count):
        m.addConstr(
            gp.quicksum(x[i, j] for i in range(node_count) if i != j) == 1,
            name=f"in_{j}"
        )

    # Enable Lazy Constraints
    m.Params.LazyConstraints = 1

    def subtour(selected, n):
        """
        Finds the shortest subtour (loop) from the selected edges.
        selected: List of (i, j) tuples representing edges where x[i, j] = 1
        """

        visited = [False] * n
        cycles = []
        for i in range(n):
            if not visited[i]:
                cycle = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    # Find the next node in the cycle
                    for (ii, jj) in selected:
                        if ii == current:
                            current = jj
                            break
                cycles.append(cycle)
        # Return the shortest cycle found
        shortest = min(cycles, key=len)
        return shortest

    def callback(model, where):
        """
        Callback function to add Lazy Constraints for subtour elimination.
        """

        if where == GRB.Callback.MIPSOL:
            # Extract the selected edges from the current solution
            selected = []
            for i, j in x.keys():
                val = model.cbGetSolution(x[i, j])
                if val > 0.5:  # Edge is included in the solution
                    selected.append((i, j))

            # Find the shortest subtour in the current solution
            tour = subtour(selected, node_count)

            # If a subtour exists that does not include all nodes, add a Lazy Constraint
            if len(tour) < node_count:
                expr = gp.quicksum(x[i, j] for i in tour for j in tour if i != j)
                model.cbLazy(expr <= len(tour) - 1)

    # Set time limit for optimization (default: 600 seconds)
    m.Params.TimeLimit = time_limit

    # Run the optimization with the callback function
    m.optimize(callback)

    if m.status in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT]:
        # Extract selected edges from the optimal solution
        selected = [(i, j) for i, j in x.keys() if x[i, j].X > 0.5]

        # Reconstruct the tour starting from node 0
        tour = [0]
        current = 0
        while len(tour) < node_count:
            for i, j in selected:
                if i == current:
                    tour.append(j)
                    current = j
                    break

        return tour, m.ObjVal  # Return the computed tour and objective value (total distance)
    else:
        return [], float('inf')  # Return an empty tour if no solution is found


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

    # # Choose initial solution strategy based on problem size
    # if node_count <= 500:
    #     print("Using Random Initialization (No Clustering)")
    #     initial_solution = list(range(node_count))  # Generate a random solution
    #     random.shuffle(initial_solution)

    # elif node_count <= 10_000:
    #     print("Using Clustering-Based Initialization")
    #     initial_solution = generate_initial_solution(points, num_clusters=10)  # Cluster-based initial solution

    # else:
    #     print("Using Greedy Local Search Initialization (Large Problem)")
    #     best_solution, best_distance = greedy_nearest_neighbor(points)  # Use greedy heuristic for large problems
    #     return f'{best_distance:.2f} 0\n' + ' '.join(map(str, best_solution))

    # # Choose initial solution strategy based on problem size
    # if node_count <= 3000:
    #     print("Using 2-OPT Tabu Search")
    #     initial_solution = list(range(node_count))  # Generate a random solution
    #     random.shuffle(initial_solution)

    #     # Run 2-OPT Tabu Search for problems with ≤ 3,000 nodes
    #     best_solution, best_distance = tabu_search(points, node_count, initial_solution)
    #     return f'{best_distance:.2f} 0\n' + ' '.join(map(str, best_solution))
    
    # else:
    #     print("Using Greedy Local Search Initialization (Large Problem)")

    #     # Use greedy heuristic for large problems
    #     best_solution, best_distance = greedy_nearest_neighbor(points)  
    #     return f'{best_distance:.2f} 0\n' + ' '.join(map(str, best_solution))
    
    # Case 1: Use Gurobi solver for small problems
    if node_count <= 300:
        print("Using Gurobi MILP Solver for TSP")

        # Solving  TSP using Gurobi optimizer
        best_solution, best_distance = gurobi_solver(points, node_count)  
        return f'{best_distance:.2f} 0\n' + ' '.join(map(str, best_solution))

    # Case 2: Use 2-OPT Tabu Search for medium-sized problems
    elif node_count <= 1500:
        print("Using 2-OPT Tabu Search")
        initial_solution = list(range(node_count))  # Generate a random initial solution
        random.shuffle(initial_solution)

        # Solve TSP using Tabu Search
        best_solution, best_distance = tabu_search(points, node_count, initial_solution)
        return f'{best_distance:.2f} 0\n' + ' '.join(map(str, best_solution))
    
    # Case 3: Use Greedy Nearest Neighbor for large problems
    else:
        print("Using Greedy Local Search for Large Problem")

        # Solve TSP using a greedy heuristic
        best_solution, best_distance = greedy_nearest_neighbor(points)  
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
