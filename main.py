from testbed import TestBed
from tsp_solvers import one_shot_exact, iterative_exact, alternate_matching, Tsp, lk_tsp
import numpy as np

if __name__ == "__main__":
    testbed = TestBed('testcases')
    # testbed.test(iterative_exact)
    testbed.test(alternate_matching)
    # testbed.test(lk_tsp)
    # testbed.test(lk_tsp)
    # t = Tsp('./tc5.txt')
    # t.getHeuristicSolution()
    # adj = np.asarray(
    #     [
    #     [0 ,2,3,4,5,8,11],
    #     [2,0,6,7,8,5,7],
    #     [3,6,0,9,10,7,5],
    #     [4,7,9,0,11,5,7],
    #     [5,8,10,11,0,7,5],
    #     [8,5,7,5,7,0,2],
    #     [11,7,5,7,5,2,0]
    #     ]
    # )
    # alternate_matching(adj)
    # lk_tsp(adj)

