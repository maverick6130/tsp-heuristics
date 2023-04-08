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

def iterative_exact(A):
    N = A.shape[0]

    indices = [(i,j) for i in range(N) for j in range(N) if i!=j]
    model = ConcreteModel()
    solver = SolverFactory('glpk')
    model.x = Var(indices, domain=Binary)
    model.objective = Objective(
        expr = sum([A[idx[0]][idx[1]]*model.x[idx] for idx in indices]), 
        sense=minimize
    )
    model.constraints = ConstraintList()
    for i in range(N):
        model.constraints.add(
            sum([
                model.x[(i,j)] for j in range(N) if j != i
            ]) == 1
        )
        model.constraints.add(
            sum([
                model.x[(j,i)] for j in range(N) if j != i
            ]) == 1
        )

    def check(model):
        cycles = []
        nodes = set(range(N))
        nxt = [-1 for _ in range(N)]
        cost = 0
        for i in indices:
            if model.x[i]() > 0:
                nxt[i[0]] = i[1]
                cost += A[i[0]][i[1]]
            
        while len(nodes):
            cyc = [nodes.pop()]
            cur = cyc[0]
            while nxt[cur] in nodes:
                cyc.append(nxt[cur])
                nodes.remove(nxt[cur])
                cur = nxt[cur]
            cycles.append(cyc)
        if len(cycles) == 1:
            sol = cycles[0]
            return True
        S = set()
        for c in cycles[:-1]:
            S.update(set(c))
            model.constraints.add(
                sum([
                    model.x[idx] for idx in indices if idx[0] in S and idx[1] not in S
                ]) >= 1
            )
        return False
    
    solver.solve(model)
    while not check(model):
        solver.solve(model)

    return value(model.objective)