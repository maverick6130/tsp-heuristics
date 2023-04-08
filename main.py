from testbed import TestBed
from tsp_solvers import one_shot_exact, iterative_exact, alternate_matching

if __name__ == "__main__":
    testbed = TestBed('testcases')
    testbed.test(alternate_matching)