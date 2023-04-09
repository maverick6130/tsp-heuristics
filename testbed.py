import tsplib95
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
import concurrent.futures
import signal
import os
import json
import time

MAXSIZE = 1000
MAXTIME = 500
class TestBed:
    def __init__(self, tc_folder):
        self.max_subprocess = os.cpu_count()

        with open(os.path.join(tc_folder, 'testcases.json')) as f:
            self.solution = json.load(f)

        self.problems = list(self.solution.keys())
        self.problem_size = { prob : len(list(tsplib95.load(os.path.join(tc_folder, f'{prob}.tsp')).get_nodes()))
                         for prob in self.problems }
        self.problems = [ prob for prob in self.problems if self.problem_size[prob] <= MAXSIZE ]
        self.adj_mtx = { prob : nx.to_numpy_array(tsplib95.load(os.path.join(tc_folder, f'{prob}.tsp')).get_graph())
                    for prob in tqdm(self.problems, desc='Loading testcases') }
    
    def timer_based_wrapper(solver, instance, index):
        print(f'Executing {solver.__name__} of size {instance.shape[0]} | ID : {index}')
        def handler(signum, frame):
            raise Exception("TIMEOUT")
        signal.signal(signal.SIGALRM, handler)
        start = time.perf_counter_ns()  
        signal.alarm(MAXTIME)
        try:
            output = solver(instance)
        except Exception as e:
            print(e)
            output = None
        signal.alarm(0)
        end = time.perf_counter_ns()
        print(f'ID {index} completed')
        return index, output, (end - start)/1e9

    def test(self, tsp_solver):
        sizes = []
        optim_res = []
        perf_res = []
        print(f'Testing {tsp_solver.__name__}')

        with concurrent.futures.ProcessPoolExecutor(self.max_subprocess) as executor:
            futures = [executor.submit(TestBed.timer_based_wrapper, tsp_solver, self.adj_mtx[prob], i)\
                    for i, prob in enumerate(self.problems)]

        for future in concurrent.futures.as_completed(futures):
            idx, sol, te = future.result()
            prob = self.problems[idx]
            print(f'Problem : {prob} | Output : {sol} | Time Elapsed : {te}')

            if sol is not None:
                sizes.append(self.problem_size[prob])
                optim_res.append(sol/self.solution[prob])
                perf_res.append(te)

        index_arr = [ idx for _, idx in sorted([(size, idx) for idx, size in enumerate(sizes)]) ]
        
        def order_by_idx_arr(arr, idx_arr):
            return [ arr[i] for i in idx_arr ]
        sizes = order_by_idx_arr(sizes, index_arr)
        optim_res = order_by_idx_arr(optim_res, index_arr)
        perf_res = order_by_idx_arr(perf_res, index_arr)

        plt.figure()
        plt.plot(sizes, optim_res)
        plt.title(tsp_solver.__name__)
        plt.xlabel('Instance Size')
        plt.ylabel('Result : Optimal Result')
        plt.savefig(f'{tsp_solver.__name__}_optim.png')

        plt.figure()
        plt.plot(sizes, perf_res)
        plt.title(tsp_solver.__name__)
        plt.xlabel('Instance Size')
        plt.ylabel('Time (s)')
        plt.savefig(f'{tsp_solver.__name__}_perf.png')

        with open(f'{tsp_solver.__name__}_optim.json', 'w') as f:
            json.dump({'sizes' : sizes, 'optim_res' : optim_res, 'perf_res':perf_res}, f)