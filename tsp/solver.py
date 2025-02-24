#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
import time

from collections import namedtuple, deque
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def total_distance(tour, points):
    distance = 0
    for i in range(len(tour)):
        distance += length(points[tour[i]], points[tour[(i + 1) % len(tour)]])
    return distance

def two_opt_swap(route, i, k):
    return route[:i] + list(reversed(route[i:k+1])) + route[k+1:]

def tabu_search(points, node_count, iterations=1000, tabu_size=100, time_limit=1800):
    start_time = time.time()
    current_solution = list(range(node_count))
    random.shuffle(current_solution)
    best_solution = list(current_solution)
    best_distance = total_distance(best_solution, points)
    
    tabu_list = deque(maxlen=tabu_size)
    
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
        
        neighborhood.sort(key=lambda x: x[1])
        best_neighbor, best_neighbor_distance, move = neighborhood[0]
        
        if best_neighbor_distance < best_distance:
            best_solution = best_neighbor
            best_distance = best_neighbor_distance
        
        current_solution = best_neighbor
        tabu_list.append(move)
    
    return best_solution, best_distance

def ortools_tsp(points, node_count):
    manager = pywrapcp.RoutingIndexManager(node_count, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(length(points[from_node], points[to_node]) * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.FromSeconds(120)  # Set time limit to 120 seconds
    
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        route_distance = 0
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        route.append(manager.IndexToNode(index))
        return route, route_distance / 1000  # Convert back to original scale
    else:
        return [], float('inf')


def solve_it(input_data):
    lines = input_data.strip().split('\n')
    node_count = int(lines[0])
    points = []
    for i in range(1, node_count + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # Optimization Problem
    # # visit the nodes in the order they appear in the file
    # solution = range(0, nodeCount)

    # # calculate the length of the tour
    # obj = length(points[solution[-1]], points[solution[0]])
    # for index in range(0, nodeCount-1):
    #     obj += length(points[solution[index]], points[solution[index+1]])

    # Tabu Search
    best_solution, best_distance = tabu_search(points, node_count)

    # OR-Tools
    # best_solution, best_distance = ortools_tsp(points, node_count)

    output_data = f'{best_distance:.2f} 0\n'
    output_data += ' '.join(map(str, best_solution))
    return output_data

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

