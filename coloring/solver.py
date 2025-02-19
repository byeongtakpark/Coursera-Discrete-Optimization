#!/usr/bin/python
# -*- coding: utf-8 -*-
from ortools.sat.python import cp_model 

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # build ortools model
    m = cp_model.CpModel()
    
    # Vars : c represents the color of i
    colors = [m.NewIntVar(0, node_count-1, f'c_{i}') for i in range(node_count)] 

    # Constraints
    for i, j in edges:
        m.Add(colors[i] != colors[j])
    
    # Objective: Minimmize the maximum color used
    max_color = m.NewIntVar(0, node_count-1, 'max_color')
    for c in colors:
        m.Add(c <= max_color)

    m.Minimize(max_color)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300  # Set a time limit
    status = solver.Solve(m)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solution = [solver.Value(colors[i]) for i in range(node_count)]
        opt = 1 if status == cp_model.OPTIMAL else 0
        output_data = f"{solver.Value(max_color) + 1} {opt}\n" + " ".join(map(str, solution))
    
    else:
        output_data = "No solution found."
    
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

