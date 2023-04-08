from testbed import TestBed
from tsp_solvers import one_shot_exact

if __name__ == "__main__":
    testbed = TestBed('testcases')
    testbed.test(one_shot_exact)