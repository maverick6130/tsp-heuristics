import json
import numpy as np
from pyomo.environ import *

import numpy as np


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




def lk_tsp(dist_matrix):
    n = len(dist_matrix)
    # Create a tour using the nearest neighbor heuristic
    tour = nearest_neighbor(dist_matrix)
    best_tour = tour
    best_cost = tour_cost(tour, dist_matrix)
    improve = True
    while improve:
        improve = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Get the candidate edges
                a, b, c, d = candidate_edges(tour, i, j)
                if c < 0:
                    continue
                # Perform the Lin-Kernighan move
                new_tour = lin_kernighan(tour, a, b, c, d)
                new_cost = tour_cost(new_tour, dist_matrix)
                if new_cost < best_cost:
                    best_tour = new_tour
                    best_cost = new_cost
                    improve = True
        tour = best_tour
    return best_cost

def nearest_neighbor(dist_matrix):
    n = len(dist_matrix)
    unvisited = list(range(1, n))
    tour = [0]
    while unvisited:
        nearest = min(unvisited, key=lambda x: dist_matrix[tour[-1]][x])
        tour.append(nearest)
        unvisited.remove(nearest)
    tour.append(0)
    return tour

def tour_cost(tour, dist_matrix):
    cost = 0
    for i in range(len(tour) - 1):
        cost += dist_matrix[tour[i]][tour[i+1]]
    return cost

def candidate_edges(tour, i, j):
    a, b = tour[i], tour[i+1]
    c, d = tour[j], tour[(j+1)%len(tour)]
    e, f = tour[(i-1)%len(tour)], tour[(j+1)%len(tour)]
    gain = dist_matrix[e][c] + dist_matrix[f][b] - dist_matrix[e][b] - dist_matrix[f][c]
    return a, c, d, b if gain > 0 else c, d, a, f

def lin_kernighan(tour, a, b, c, d):
    if b < c:
        subtour = tour[b:c+1]
        subtour.reverse()
        return tour[:b] + subtour + tour[c+1:]
    else:
        subtour = tour[d:b+1]
        subtour.reverse()
        return tour[:d+1] + subtour + tour[b+1:]


