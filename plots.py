import matplotlib.pyplot as plt
import json
import pandas as pd



algos = {
    'DFJ': {},
    'SD_modified': {},
    'LK': {},
    'SD': {}
}
algos['DFJ'] = json.load(open('iterative_exact_optim.json'))
algos['SD_modified'] = json.load(open('alternate_matching_optim.json'))
algos['LK'] = json.load(open('lk_tsp_optim.json'))
algos['SD'] = json.load(open('alternate_matching_orig_optim.json'))

algos['DFJ']['optim_res'] = [1 for _ in range(len(algos['LK']['sizes']))]
algos['DFJ']['perf_res'] += [None for _ in range(len(algos['LK']['sizes']) - len(algos['DFJ']['perf_res']))]
df = pd.DataFrame(None, index = range(len(algos['LK']['sizes'])), columns= ['size'] + [f'{a}_{f}' for a in algos for f in ['optimality', 'performance']])

df['size'] = algos['LK']['sizes']
for algo in algos:
    df[f'{algo}_optimality'] = algos[algo]['optim_res']
    df[f'{algo}_performance'] = algos[algo]['perf_res']

df['bin'] = df['size'] - (df['size']%30);
# print(df)
df = df.groupby('bin').mean()
df['DFJ_performance'].fillna(450, inplace=True)


fig = df.plot(y=['DFJ_optimality', 'SD_modified_optimality', 'LK_optimality', 'SD_optimality'], kind='line', figsize=(10,10))
fig.set_xlabel('Size of the problem')

fig.set_xscale('log')
fig.grid(True)
fig.set_ylabel('Optimality ratio')
fig.set_title('Optimality ratio of the algorithms')
fig.get_figure().savefig('optimality.png')



fig = df.plot(y=['DFJ_performance', 'SD_modified_performance', 'LK_performance', 'SD_performance'], kind='line', figsize=(10,10))
fig.set_xlabel('Size of the problem')
fig.set_ylabel('Time (s)')
fig.grid(True)
fig.set_title('Performance of the algorithms')
fig.get_figure().savefig('performance.png')

