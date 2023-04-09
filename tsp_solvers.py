import json
import numpy as np
from pyomo.environ import *

from pyomo.environ import *
import random
import time
TMAX = 300
class Tsp:
    def __init__(self, A):
        self.base = "greedy"  # "random" or "greedy"
        self.numHeuristics = 1 # 1 or 2
        self.sol = []
        self.cost = 0
        self.adj = A.T.tolist()
        self.origN = len(self.adj)
        self.extra = 6 - len(self.adj) % 6
        if self.extra == 6:
            self.extra = 0
        self.extraEdge = 1 + A.max()
        self.n = len(self.adj)
        # print(self.adj)
        for _ in range(self.extra):
            self.adj.append([self.extraEdge for _ in range(self.n - 1)])
            self.adj[-1] += [0 for _ in range(self.extra + 1)]
        for i in range(self.n - 1):
            self.adj[i] += [self.extraEdge for _ in range(self.extra)]
        self.adj[self.n - 1] += [0 for _ in range(self.extra)]
        self.n += self.extra


        

    # def __init__(self, file):
    #     self.base = "greedy"  # "random" or "greedy"
    #     self.numHeuristics = 2 # 1 or 2

    #     with open(file, 'r') as f:
    #         self.adj = [
    #             [int(x) for x in l.strip().split(' ')]
    #             for l in f.readlines()
    #         ]
    #     self.n = len(self.adj)
    #     # print(self.adj)
    #     while self.n % 6:
    #         self.adj.append([0 for _ in range(self.n)])
    #         self.n += 1
    #         # print(self.adj)
    #         for i in range(self.n):
    #             # print(self.adj[i])
    #             self.adj[i].append(0)

    #     self.sol = []
    #     self.cost = 0

    def getCorrectSolution(self):

        indices = [(i,j) for i in range(self.n) for j in range(self.n) if i!=j]
        model = ConcreteModel()
        solver = SolverFactory('glpk')
        model.x = Var(indices, domain=Binary)
        model.objective = Objective(
            expr = sum([self.adj[idx[0]][idx[1]]*model.x[idx] for idx in indices]), 
            sense=minimize
        )
        model.constraints = ConstraintList()
        for i in range(self.n):
            model.constraints.add(
                sum([
                    model.x[(i,j)] for j in range(self.n) if j != i
                ]) == 1
            )
            model.constraints.add(
                sum([
                    model.x[(j,i)] for j in range(self.n) if j != i
                ]) == 1
            )

        def check(model):
            cycles = []
            nodes = set(range(self.n))
            nxt = [-1 for _ in range(self.n)]
            self.cost = 0
            for i in indices:
                if model.x[i]() > 0:
                    nxt[i[0]] = i[1]
                    self.cost += self.adj[i[0]][i[1]]
                
            while len(nodes):
                cyc = [nodes.pop()]
                cur = cyc[0]
                while nxt[cur] in nodes:
                    cyc.append(nxt[cur])
                    nodes.remove(nxt[cur])
                    cur = nxt[cur]
                cycles.append(cyc)
            if len(cycles) == 1:
                self.sol = cycles[0]
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


    def getBaseSolution(self):
        if self.base == "random":
            self.sol = (list(range(self.n)))
            random.shuffle(self.sol)
            self.cost = 0
            for i in range(self.n):
                self.cost += self.adj[self.sol[i]][self.sol[(i+1)%self.n]]

        elif self.base == "greedy":
        
            vist = [False for _ in range(self.n)]
            self.sol = [0]
            vist[0] = True
            self.cost = 0
            while len(self.sol) != self.n:
                cur = vist[-1]
                dist = [self.adj[cur][i] if not vist[i] else float('inf') for i in range(self.n)]
                self.cost += min(dist)
                self.sol.append(dist.index(min(dist)))
                vist[self.sol[-1]] = True

            self.cost += self.adj[self.sol[-1]][0]


    def reduceSolution(self):
        holes = self.sol[::2]
        # print(holes)
        reducedPath = self.sol[1::2]
        reducedPath.append(reducedPath[0])
        indices = [(i,(x,y)) for i in holes for (x,y) in zip(reducedPath[:-1], reducedPath[1:])]

        model = ConcreteModel()
        solver = SolverFactory('glpk')
        model.x = Var(indices, domain=Binary)
        model.objective = Objective(
            expr = sum([model.x[idx]*(self.adj[idx[1][0]][idx[0]] + self.adj[idx[0]][idx[1][1]]) for idx in indices]),
            sense=minimize
        )

        model.constraints = ConstraintList()
        for i in holes:
            model.constraints.add(
                sum([
                    model.x[(i,(x,y))] for (x,y) in zip(reducedPath[:-1], reducedPath[1:])
                ]) == 1
            )
        for (x,y) in zip(reducedPath[:-1], reducedPath[1:]):
            model.constraints.add(
                sum([
                    model.x[(i,(x,y))] for i in holes
                ]) == 1
            )

        solver.solve(model)
        self.sol = []
        for i,x in enumerate(reducedPath[:-1]):
            for j in holes:
                if model.x[(j,(x,reducedPath[i+1]))]() > 0:
                    self.sol += [x, j]
                    break
        self.cost = 0
        for i in range(len(self.sol)-1):
            self.cost += self.adj[self.sol[i]][self.sol[i+1]]
        self.cost += self.adj[self.sol[-1]][self.sol[0]]



    def reduceSolution2(self):
        holes = [(self.sol[i], self.sol[i+1]) for i in range(0, self.n, 3)]
        reducedPath = self.sol[2::3]
        reducedPath.append(reducedPath[0])
        # print("holes", holes)
        # print("reducedPath", reducedPath)
        indices = [(i,(x,y)) for i in holes for (x,y) in zip(reducedPath[:-1], reducedPath[1:])]
        # print("indices")
        # for x in indices:
        #     print(x, end="  |  ")
        model = ConcreteModel()
        solver = SolverFactory('glpk')
        model.x = Var(indices, domain=Binary)
        model.objective = Objective(
            expr = sum(
                [
                    model.x[idx]*(self.adj[idx[1][0]][idx[0][0]] + self.adj[idx[0][0]][idx[0][1]] + self.adj[idx[0][1]][idx[1][1]])
                    for idx in indices
                ]
            )
            ,sense=minimize)
        model.constraints = ConstraintList()
        for i in holes:
            model.constraints.add(
                sum([
                    model.x[(i,(x,y))] for (x,y) in zip(reducedPath[:-1], reducedPath[1:])
                ]) == 1
            )
        for (x,y) in zip(reducedPath[:-1], reducedPath[1:]):
            model.constraints.add(
                sum([
                    model.x[(i,(x,y))] for i in holes
                ]) == 1
            )
        
        solver.solve(model)

        self.sol = []
        for i,x in enumerate(reducedPath[:-1]):
            for j in holes:
                if model.x[(j,(x,reducedPath[i+1]))]() > 0:
                    self.sol +=  [x, j[0], j[1]]
                    break

        for i in range(0,self.n, 3):
            a,b,c,d = self.sol[i], self.sol[i+1], self.sol[i+2], self.sol[(i+3)%self.n]
            if self.adj[a][b] + self.adj[b][c] + self.adj[c][d] > self.adj[a][c] + self.adj[c][b] + self.adj[b][d]:
                self.sol[i+1], self.sol[i+2] = self.sol[i+2], self.sol[i+1]
        self.cost = 0

        for i in range(len(self.sol)-1):
            self.cost += self.adj[self.sol[i]][self.sol[i+1]]
        self.cost += self.adj[self.sol[-1]][self.sol[0]]


    def getActualCost(self):
        
       
        sol = [x for x in self.sol if x < self.origN]
        self.cost = 0
        for i in range(len(sol)):
            self.cost += self.adj[sol[i]][sol[(i+1)%len(sol)]]
        sol.sort()
        if sol != [x for x in range(self.origN)]:
            print("ERROR!!!!")
        # self.cost -= self.extraEdge
        # origN = self.n - self.extra
        # idx = self.sol.index(origN-1)
        # idx1, idx2 = idx, idx
        # while(self.sol[idx1] >= origN - 1):
        #     idx1 -= 1
        #     if idx1 < 0:
        #         idx1 = self.n - 1
        
        # while(self.sol[idx2] >= origN - 1):
        #     idx2 += 1
        #     if idx2 == self.n:
        #         idx2 = 0
        
        # if((idx1 + 1)% self.n == idx):
        #     self.cost += self.adj[self.sol[idx]][self.sol[idx2]]
        # else:
        #     self.cost += self.adj[self.sol[idx1]][self.sol[idx]]



    def getHeuristicSolution(self, delta=1):
        
        self.getBaseSolution()
        prevCost = self.cost
        self.reduceSolution()
        tStart = time.time()
        while True:
            if time.time() - tStart > TMAX:
                break
            # if self.numHeuristics == 2:
            self.reduceSolution() 
            # self.reduceSolution2()
            

            if self.cost < prevCost:
                prevCost = self.cost
            else:
                break


    def getSolution(self, type, delta=1):
        if type == "heuristic":
            self.getHeuristicSolution(delta)
        elif type == "base":
            self.getBaseSolution()
        else:
            self.getCorrectSolution()
        self.getActualCost()
        return self.cost
    


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

def alternate_matching_orig(A):
    prob = Tsp(A)
    return prob.getSolution("heuristic")
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
    tStart = time.time()
    improve = True
    while improve:
        if time.time() - tStart > TMAX:
            break
        improve = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Get the candidate edges
                a, b, c, d = candidate_edges(tour, i, j, dist_matrix)
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
    for i in range(len(tour)):
        cost += dist_matrix[tour[i]][tour[(i+1)%len(tour)]]
    return cost

def candidate_edges(tour, i, j, dist_matrix):
    a, b = tour[i], tour[i+1]
    c, d = tour[j], tour[(j+1)%len(tour)]
    e, f = tour[(i-1)%len(tour)], tour[(j+1)%len(tour)]
    gain = dist_matrix[e][c] + dist_matrix[f][b] - dist_matrix[e][b] - dist_matrix[f][c]
    if gain > 0:
        return a,c,d,b 
    return c,d,a,f
    # return (a, c, d, b if gain > 0 else c, d, a, f)

def lin_kernighan(tour, a, b, c, d):
    if b < c:
        subtour = tour[b:c+1]
        subtour.reverse()
        return tour[:b] + subtour + tour[c+1:]
    else:
        subtour = tour[d:b+1]
        subtour.reverse()
        return tour[:d+1] + subtour + tour[b+1:]


