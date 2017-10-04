import pandas
import numpy

import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

names = ['pretraining', 'main', 'annealing1', 'annealing2']

def get_df(path):
    dfs = []
    for name in names:
        with open(path + name + '_df.pkl') as f:
            dfs.append(cPickle.load(f))
    return pandas.concat(dfs).reset_index(), [len(dfs[0]), len(dfs[1]), len(dfs[2]), len(dfs[3])]

df_all, lengths = get_df('./')
df_baseline, _ = get_df('../wsj_paper1_test/')

def plotline(line, *args, **kwargs):
    return plt.plot(line.dropna().index, line.dropna(), *args, **kwargs)


plt.figure(figsize=(10, 8))
our_train, = plotline(df_all['costs_forward_aux'].rolling(1000).mean().dropna()[::500], 'orange', linestyle='dotted', linewidth=3, alpha=0.9)
bl_train, = plotline(df_baseline['train_cost'].rolling(1000).mean().dropna()[::500], 'red', linestyle='dotted', linewidth=3, alpha=0.9)
l2_cost, = plotline(df_all['l2_cost_aux'].rolling(1000).mean().dropna()[::500], 'y', linewidth=3, alpha=0.9)
our_valid, = plotline(df_all['valid_costs_forward_aux'], 'orange', linewidth=3)
bl_valid, = plotline(df_baseline['valid_train_cost'], 'red', linewidth=3)
plt.legend([our_train, bl_train, our_valid, bl_valid, l2_cost],
           ['TwinNet train', 'Baseline train', 'TwinNet valid', 'Baseline valid', 'L2 cost'],
           prop={'size': 20})
plt.grid(which='both')
plt.yscale('log')
plt.xlabel('iterations', fontsize=16)
plt.ylabel('cost', fontsize=16)
plt.axvline(x=lengths[0], color='k', linestyle='dotted')
plt.axvline(x=sum(lengths[:2]), color='k', linestyle='dotted')
plt.axvline(x=sum(lengths[:3]), color='k', linestyle='dotted')
plt.savefig('test.eps', bbox_inches='tight')
plt.close()



def plot_l2(ex=0):
    plt.figure(figsize=(10, 2))
    n = int(dt['labels_mask'][:, ex].sum())
    plt.plot(cost2[1:n, ex])
    transcript = [dataset.num2char[c] if len(dataset.num2char[c]) == 1 else ' ' for c in dt['labels'][:n, ex]]
    plt.xticks(range(1, n), transcript, fontsize=8, fontname='DejaVu Sans Mono')
    plt.grid(axis='y')
    for pos in [pos for pos, c in enumerate(transcript) if c == ' ']:
        plt.axvline(x=pos + 1, color='k', linestyle='dotted')
    plt.ylabel('cost', fontsize=12)
    plt.savefig('l2cost-{}-small.eps'.format(ex+1), bbox_inches='tight')
    plt.close()

from blocks import serialization
from blocks.filter import VariableFilter
import theano
with open('wsj_paper/annealing2.zip') as f:
    main_loop = serialization.load(f)

costs = VariableFilter(theano_name='l2_cost_aux')(main_loop.model)
f = theano.function(main_loop.model.inputs, costs)

stream = main_loop.data_stream
while hasattr(stream, 'data_stream'):
    stream = stream.data_stream
dataset = stream.dataset
it = main_loop.data_stream.get_epoch_iterator(as_dict=True)
dt = next(it)
cost1, cost2 = f(**dt)
plot_l2(0)
plot_l2(1)
import IPython
IPython.embed()
