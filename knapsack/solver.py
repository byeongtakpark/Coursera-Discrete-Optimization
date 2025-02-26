#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import namedtuple
from ortools.linear_solver import pywraplp

Item = namedtuple("Item", ['index', 'value', 'weight'])

def parse_input(input_data):
    # Parse the input data by splitting it into lines
    lines = input_data.split('\n')

    # Extract the first line to get item count and capacity
    firstLine = lines[0].split()
    item_count = int(firstLine[0])  # Number of items
    capacity = int(firstLine[1])  # Capacity

    items = []

    # Loop through each subsequent line to create Item objects
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))  # Create an Item object and append to the list
    
    # Return the item count, capacity, and the list of items
    return (item_count, capacity, items)

def solve_it(input_data):
    item_count, capacity, items = parse_input(input_data)

    # value, taken = solve_it_dp(item_count, capacity, items)
    value, taken = solve_it_ortools(item_count, capacity, items)
    
    # Prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def solve_it_dp(item_count, capacity, items):
    # Dynamic Programming approach to solve the knapsack problem
    value = 0  # Initialize total value of the selected items
    taken = [0] * item_count  # Array to track which items are selected
    
    # Create a DP table to store the maximum value for each item and capacity combination
    dp = [[0] * (capacity+1) for _ in range(item_count+1)]
    
    # Fill the DP table
    for i in range(1, item_count+1):
        for j in range(1, capacity+1):
            # If the current item can fit in the knapsack
            if items[i-1].weight <= j:
                # Take the maximum of not including or including the item
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-items[i-1].weight] + items[i-1].value)
            else:
                # Otherwise, do not include the item
                dp[i][j] = dp[i-1][j]
    
    # The total maximum value is stored at dp[item_count][capacity]
    value = dp[item_count][capacity]
    
    # Backtracking to find the selected items
    w = capacity
    for i in range(item_count, 0, -1):
        if dp[i][w] != dp[i-1][w]:  # If the value differs, the item was included
            taken[i-1] = 1  # Mark the item as taken
            w -= items[i-1].weight  # Reduce the remaining capacity
    
    return int(value), taken  # Return the total value and the list of selected items

def solve_it_ortools(item_count, capacity, items):    
    # Initialize the linear solver (using the CBC Solver)
    solver = pywraplp.Solver.CreateSolver('SCIP')

    if not solver:
        print('Solver not created!')
        return None, None
    
    # Declare variables: x[i] represents whether item i is selected (0 or 1)
    x = []
    for i in range(item_count):
        x.append(solver.IntVar(0.0, 1.0, f'x_{i}'))
    
    # Add constraint: the total weight of selected items must not exceed the capacity
    solver.Add(solver.Sum([x[i] * items[i].weight for i in range(item_count)]) <= capacity)

    # Define the objective function: maximize the total value of selected items
    objective = solver.Objective()
    for i in range(item_count):
        objective.SetCoefficient(x[i], items[i].value)
    objective.SetMaximization()
    
    # Solve the problem
    status = solver.Solve()

    # Process the result
    if status == pywraplp.Solver.OPTIMAL:
        best_value = 0  # Initialize the best value
        best_taken = [0] * item_count  # Initialize the list to track selected items
        for i in range(item_count):
            if x[i].solution_value() > 0.5:  # If the item is selected (close to 1)
                best_taken[i] = 1
                best_value += items[i].value  # Add the value of the selected item
        return int(best_value), best_taken  # Return the total value and the selected items
    else:
        print('The problem does not have an optimal solution.')
        return 0, [0] * item_count  # Return 0 value and no items selected if no optimal solution

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

# !/usr/bin/python
# -*- coding: utf-8 -*-