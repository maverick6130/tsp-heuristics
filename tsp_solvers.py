import json
import numpy as np
from pyomo.environ import *

from pyomo.environ import *
import random
import time
class Tsp:
    def __init__(self, A):
        self.base = "greedy"  # "random" or "greedy"
        self.numHeuristics = 2 # 1 or 2

        self.adj = A.T.tolist()
        self.extra = 0
        self.extraEdge = 1 + A.max()
        self.n = len(self.adj)
        # print(self.adj)
        while self.n % 6:
            self.extra += 1
            self.adj.append([self.extraEdge for _ in range(self.n)])
            self.n += 1
            # print(self.adj)
            for i in range(self.n - 1):
                # print(self.adj[i])
                self.adj[i].append(self.extraEdge)
            self.adj[-1].append(0)

        self.sol = []
        self.cost = 0

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

    def validate(self):
        x = self.sol
        x.sort()
        if x != [t for t in range(self.n)]:
            print(x)
            print([t for t in range(self.n)])
            return False
        return True

    def getActualCost(self):
        if self.extra == 0:
            return
        self.cost -= self.extraEdge
        origN = self.n - self.extra
        idx = self.sol.index(origN-1)
        idx1, idx2 = idx, idx
        while(self.sol[idx1] >= origN - 1):
            idx1 -= 1
            if idx1 < 0:
                idx1 = self.n - 1
        
        while(self.sol[idx2] >= origN - 1):
            idx2 += 1
            if idx2 == self.n:
                idx2 = 0
        
        if((idx1 + 1)% self.n == idx):
            self.cost += self.adj[self.sol[idx]][self.sol[idx2]]
        else:
            self.cost += self.adj[self.sol[idx1]][self.sol[idx]]



    def getHeuristicSolution(self, delta=1):
        
        self.getBaseSolution()
        prevCost = self.cost
        self.reduceSolution()
        
        while True:
            if not self.validate():
                print("ERROR")
                
            self.reduceSolution2()
            if self.numHeuristics == 2:
                self.reduceSolution()

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
        if not self.validate():
            print("ERROR")
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

def alternate_matching(A):
    prob = Tsp(A)
    return prob.getSolution("heuristic")