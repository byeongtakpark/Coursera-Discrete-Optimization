# Send More Money
#
# Task : Assign differents digits to letters to satisfy the addition 
# 
#        S  E  N  D 
# +      M  O  R  E
# ------------------
#    M   O  N  E  Y 

class Solver:
    def __init__(self):
        'Defines the problem variables (letters in "SENDMORY") and assigns them a possible domain (digits 0-9).'
        self.variables = {i: None for i in "SENDMORY"} 
        self.domain = list(range(0, 10)) # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

    def is_valid(self):
        'Checks if the current digit assignments satisfy the equation SEND + MORE = MONEY.'
        if self.variables['S'] == 0 or self.variables['M'] == 0:
            return False
        
        send = (self.variables['S'] * 1000 + self.variables['E'] * 100 + self.variables['N'] * 10 + self.variables['D'])
        more = (self.variables['M'] * 1000 + self.variables['O'] * 100 + self.variables['R'] * 10 + self.variables['E'])
        money = (self.variables['M'] * 10000 + self.variables['O'] * 1000 + self.variables['N'] * 100 + self.variables['E'] * 10 + self.variables['Y'])
        return send + more == money

    def backtracking(self, index=0):
        'Recursively assigns digits to letters while ensuring uniqueness.'
        if index == len(self.variables): # All letters assigned
            return self.is_valid()
           
        var = list(self.variables.keys())[index]
           
        for value in self.domain: 
            if value in self.variables.values(): # Ensure unique digit assignment
                continue
            
            self.variables[var] = value
            if self.backtracking(index + 1):
                return True
            
            self.variables[var] = None # Backtrack if the assignment fails
        
        return False

    def solve(self):
        'Initiates the backtracking process.'
        if self.backtracking():
            return self.variables
        return None

solver = Solver()

solution = solver.solve()
if solution:
    print("Solution Found:", solution)

else:
    print("No solution found.")

