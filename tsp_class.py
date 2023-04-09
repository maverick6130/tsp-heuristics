from abc import ABCMeta, abstractmethod


class TSP():
    __metaclass__ = ABCMeta

    edges = {}  # Global cost matrix
    ratio = 10.  # Global ratio
    routes = {}  # Global routes costs

    def __init__(self, nodes, fast=False):
        self.nodes = nodes
        self.fast = fast

        self.initial_path = nodes
        self.initial_cost = self.pathCost(nodes)
        # Do not save the initial path as it is not optimised
        self.heuristic_path = self.initial_path
        self.heuristic_cost = self.initial_cost

    def save(self, path, cost):
        self.heuristic_path = path
        self.heuristic_cost = cost

        self.routes[str(sorted(path))] = {"path": path, "cost": cost}

    def update(self, solution):
        self.heuristic_path = [i for i in self.initial_path if i in solution]
        self.heuristic_cost = self.pathCost(self.heuristic_path)

    def __str__(self):
        out = "Route with {} nodes ({}):\n".format(
            len(self.heuristic_path), self.heuristic_cost)

        if self.heuristic_cost > 0:
            out += " -> ".join(map(str, self.heuristic_path))
            out += " -> {}".format(self.heuristic_path[0])
        else:
            out += "No current route."

        return out

    @staticmethod
    def dist(i, j):
        return TSP.edges[i][j]

    @staticmethod
    def pathCost(path):
        # Close the loop
        cost = TSP.dist(path[-1], path[0])

        for i in range(1, len(path)):
            cost += TSP.dist(path[i - 1], path[i])

        return cost

    @staticmethod
    def setRatio(ratio):
        TSP.ratio = ratio

    @staticmethod
    def setEdges(edges):
        TSP.edges = edges

    def optimise(self):
        route = str(sorted(self.heuristic_path))

        if route in self.routes:
            saved = TSP.routes[route]
            self.heuristic_path = saved["path"]
            self.heuristic_cost = saved["cost"]
        else:
            self._optimise()

        return self.heuristic_path, self.heuristic_cost

    @abstractmethod
    def _optimise(self):
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
