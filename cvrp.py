import pandas as pd, numpy as np
import time
from itertools import cycle

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import pyomo.environ as pyo
solver = pyo.SolverFactory('glpk')

t_ini = time.time()

np.random.seed(42)
N = 10
demands = np.random.randint(1, 10, size=N)
demands[0] = 0
capacity = 15
# n_vehicles = 4

coordinates = np.random.rand(N, 2)
distances = squareform(pdist(coordinates, metric="euclidean"))
distances = np.round(distances, decimals=4)

# Bin packing problem to find number of vehicles
from bpp import create_bpp

bpp = create_bpp({i: d for (i, d) in enumerate(demands)}, capacity)
solver.solve(bpp)
n_vehicles = int(bpp.obj())
print(f"{n_vehicles} vehicles are required")

# CVRP
model = pyo.ConcreteModel()

# Sets
model.V = pyo.Set(initialize=range(len(demands)))
model.A = pyo.Set(initialize=range(N))
model.K = pyo.Set(initialize=range(n_vehicles))

# Parameters
model.Q = pyo.Param(initialize=capacity)
model.d = pyo.Param(model.A, model.A, initialize={(i, j): distances[i, j] for i in model.A for j in model.A})
model.q = pyo.Param(model.V, initialize={i: d for (i, d) in enumerate(demands)})

# Variables
model.x = pyo.Var(model.A, model.A, model.K, initialize=0, within=pyo.Binary)
model.u = pyo.Var(model.A, initialize=0, within=pyo.NonNegativeIntegers)

# Constraints
# Vehicle leaves node that it enters
def vehicle_leaves_node(model, j, k):
    return sum(model.x[i,j,k] for i in model.A) == sum(model.x[j,i,k] for i in model.A)

model.vehicle_leaves_node = pyo.Constraint(model.A, model.K, rule=vehicle_leaves_node)

def node_entered_once(model, j):
    if j > 0:
        return sum(model.x[i,j,k] for k in model.K for i in model.A ) == 1
    else:
        return pyo.Constraint.Skip

model.node_entered_once = pyo.Constraint(model.A, rule=node_entered_once)

def vehicle_leaves_depot(model, k):
    return sum(model.x[0,j,k] for j in model.A if j>0) == 1

model.vehicle_leaves_depot = pyo.Constraint(model.K, rule=vehicle_leaves_depot)

def vehicle_capacity(model, k):
    return sum(model.q[j] * model.x[i,j,k] for i in model.A for j in model.A if j>0) <= model.Q

model.vehicle_capacity = pyo.Constraint(model.K, rule=vehicle_capacity)

def subtour_elimination_1(model, i, j, k):
    if (i > 0) and (j > 0) and (i != j):
        return model.u[j] - model.u[i] >= model.q[j] - model.Q * (1 - model.x[i,j,k])
    else:
        return pyo.Constraint.Skip

def subtour_elimination_2(model, i):
    if i > 0:
        return model.q[i] <= model.u[i]
    else:
        return pyo.Constraint.Skip

def subtour_elimination_3(model, i):
    if i > 0:
        return model.u[i] <= model.Q
    else:
        return pyo.Constraint.Skip

model.subtour_elimination_1 = pyo.Constraint(model.A, model.A, model.K, rule=subtour_elimination_1)
model.subtour_elimination_2 = pyo.Constraint(model.A, rule=subtour_elimination_2)
model.subtour_elimination_3 = pyo.Constraint(model.A, rule=subtour_elimination_3)

def no_travel_itself(model, i, k):
    return model.x[i,i,k] == 0

model.no_travel_itself = pyo.Constraint(model.A, model.K, rule=no_travel_itself)

# Objective
model.obj = pyo.Objective(
    expr=sum(
        model.x[i, j, k] * model.d[i, j]
        for k in model.K
        for i in model.A
        for j in model.A
    ),
    sense=pyo.minimize,
)

# Print the number of decision variables and constraints
num_decision_vars = list(model.component_data_objects(pyo.Var, active=True))
num_constraints = list(model.component_data_objects(pyo.Constraint, active=True))
print("Number of decision variables = {}".format(len(num_decision_vars)))
print("Number of constraints = {}".format(len(num_constraints)))

#  Print the status of the solved LP
status = solver.solve(model)
print("Status = {}".format(status.solver.termination_condition))
print("Optimal Value = {}".format(model.obj()))
print("\n")

for var in model.x:
    if model.x[var].value == 1:
        print(f"{var}: {(model.x[var].value)}")

def collect_tours(model):
    tours = []
    for k in model.K:
        tour = [0]
        i = 0  # Start from the depot
        while True:
            for j in model.A:
                if model.x[i, j, k].value == 1:
                    tour.append(j)
                    i = j
                    break
            if i == 0:  # If returned to depot, the tour is complete
                break
        tours.append(tour)
    return tours


# Assuming you have defined your Pyomo model and solved it already
tours = collect_tours(model)
print(tours)

elapsed_time = time.time() - t_ini
print('Total run-time: {}'.format(elapsed_time))
