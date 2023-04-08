import json
import numpy as np
from pyomo.environ import *

def one_shot_exact(A):
    N = A.shape[0]

    idx_set = [(i,j) for i in range(N) for j in range(N) if i!=j]
    model = ConcreteModel()
    solver = SolverFactory('glpk')

    model.edges = Var(idx_set, domain = Binary)

    model.flows = Var(idx_set, domain = NonNegativeReals)
    flow_deficit = [ -1 ] * N
    flow_deficit[0] = N-1

    model.objective = Objective(
        expr = sum( [A[i][j] * model.edges[(i,j)] for (i,j) in idx_set] ), 
        sense = minimize
    )

    model.constraints = ConstraintList()
    for node in range(N):
        model.constraints.add( sum([ model.edges[(node, dest)] for dest in range(N) if dest != node ]) == 1 )
        model.constraints.add( sum([ model.edges[(origin, node)] for origin in range(N) if origin != node ]) == 1 )
        model.constraints.add( 
            sum([ model.flows[(node, dest)] for dest in range(N) if dest != node ]) - 
            sum([ model.flows[(origin, node)] for origin in range(N) if origin != node ])
            == flow_deficit[node] )
    for i in range(N):
        for j in range(N):
            if i != j:
                model.constraints.add( model.flows[(i,j)] <= model.edges[(i,j)] * (N-1) )

    solver.solve(model)
    return value(model.objective)