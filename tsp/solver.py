import math
import random
import time
import sys
import gurobipy as gp

from gurobipy import GRB
from collections import deque

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def parse_input_data(input_data):
    lines = input_data.strip().split('\n')
    node_count = int(lines[0])
    points = [Point(*map(float, lines[i].split())) for i in range(1, node_count + 1)]
    return points, node_count

def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def total_distance(route, points):
    return sum(length(points[route[i]], points[route[(i + 1) % len(route)]]) for i in range(len(route)))

def two_opt_swap(route, i, k):
    return route[:i] + list(reversed(route[i:k+1])) + route[k+1:]

# Greedy Nearest Neighbor algorithm for large TSP instances
def greedy_nearest_neighbor(points):
    node_count = len(points)
    unvisited = set(range(node_count))
    current = 0  
    solution = [current]
    unvisited.remove(current)

    while unvisited:
        nearest = min(unvisited, key=lambda node: length(points[current], points[node]))
        solution.append(nearest)
        unvisited.remove(nearest)
        current = nearest  

    return solution, total_distance(solution, points)

# 2-OPT Tabu Search for TSP
def tabu_search(points, node_count, initial_solution, iterations=1000, tabu_size=100, time_limit=1800):
    current_solution = initial_solution.copy()
    tabu_list = deque(maxlen=tabu_size)  
    start_time = time.time()  

    best_solution = list(current_solution)
    best_distance = total_distance(best_solution, points)

    for _ in range(iterations):
        if time.time() - start_time > time_limit:
            break

        neighborhood = []
        for i in range(1, node_count - 1):
            for k in range(i + 1, node_count):
                if (i, k) not in tabu_list:
                    new_solution = two_opt_swap(current_solution, i, k)  
                    new_distance = total_distance(new_solution, points) 
                    neighborhood.append((new_solution, new_distance, (i, k)))

        if not neighborhood:
            continue

        neighborhood.sort(key=lambda x: x[1])
        best_neighbor, best_neighbor_distance, move = neighborhood[0]

        if best_neighbor_distance < best_distance:
            best_solution = best_neighbor
            best_distance = best_neighbor_distance

        current_solution = best_neighbor
        tabu_list.append(move)

    return best_solution, best_distance

# Gurobi MILP solver for small TSP instances
def gurobi_solver(points, node_count, time_limit=600):
    distance = {(i, j): length(points[i], points[j]) if i != j else 0 for i in range(node_count) for j in range(node_count)}

    m = gp.Model("TSP")
    m.Params.OutputFlag = 0  # Suppress solver output

    x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for i in range(node_count) for j in range(node_count) if i != j}
    m.update()

    m.setObjective(gp.quicksum(distance[i, j] * x[i, j] for (i, j) in x), GRB.MINIMIZE)

    for i in range(node_count):
        m.addConstr(gp.quicksum(x[i, j] for j in range(node_count) if i != j) == 1, name=f"out_{i}")
    for j in range(node_count):
        m.addConstr(gp.quicksum(x[i, j] for i in range(node_count) if i != j) == 1, name=f"in_{j}")

    m.Params.LazyConstraints = 1

    def subtour(selected, n):
        visited = [False] * n
        cycles = []
        for i in range(n):
            if not visited[i]:
                cycle = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    for (ii, jj) in selected:
                        if ii == current:
                            current = jj
                            break
                cycles.append(cycle)
        return min(cycles, key=len)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            selected = [(i, j) for i, j in x.keys() if model.cbGetSolution(x[i, j]) > 0.5]
            tour = subtour(selected, node_count)
            if len(tour) < node_count:
                expr = gp.quicksum(x[i, j] for i in tour for j in tour if i != j)
                model.cbLazy(expr <= len(tour) - 1)

    m.Params.TimeLimit = time_limit
    
    m.optimize(callback)

    if m.status in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT]:
        selected = [(i, j) for i, j in x.keys() if x[i, j].X > 0.5]
        tour = [0]
        current = 0
        while len(tour) < node_count:
            for i, j in selected:
                if i == current:
                    tour.append(j)
                    current = j
                    break
        return tour, m.ObjVal
    else:
        return [], float('inf')

# Main function to determine the best algorithm for TSP based on problem size
def solve_it(input_data):
    points, node_count = parse_input_data(input_data)

    if node_count <= 300:
        print("Using Gurobi MILP Solver for TSP")
        best_solution, best_distance = gurobi_solver(points, node_count)
    
    elif node_count <= 3000:
        print("Using 2-OPT Tabu Search")
        initial_solution = list(range(node_count))
        random.shuffle(initial_solution)
        best_solution, best_distance = tabu_search(points, node_count, initial_solution)

    else:
        print("Using Greedy Local Search for Large Problem")
        best_solution, best_distance = greedy_nearest_neighbor(points)

    return f'{best_distance:.2f} 0\n' + ' '.join(map(str, best_solution))

# Main script execution: Reads input file and solves TSP
if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file. Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')