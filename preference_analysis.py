import numpy as np 
import json 
import seaborn as sns 
import pylab as plt
import pandas as pd


var = "accuracy"
data = {
    "group": [],
    "value": []
}
for n in [10,25,50]:
    n = str(n)
    stress_exp = list()
    for i,fname in enumerate([f"nt_stress{n}-cleaned.json", f"stress{n}-cleaned.json", f"stress_expert-cleaned.json", f"preference{n}-cleaned.json"]):
        with open(f"json_data/{fname}", 'r') as fdata:
            stress = json.load(fdata)

        if 'expert' not in fname:
            stress_exp.append([[d for d in s[var] if d >= 0] for s in stress])
        else: stress_exp.append([[d for d in s[n][var] if d >= 0] for s in stress])


    stress_exp = {key: s for key,s in zip(['untrained', 'trained', 'expert', 'preference'], stress_exp)}

    stress_exp['gd'] = stress_exp['untrained'] + stress_exp['trained'] + stress_exp['expert']

    means = {
        key: np.mean(np.array(tab), axis=1) for key, tab in stress_exp.items()
    }

    print(len(stress_exp['preference']))
    print(stress_exp['preference'][0])

    means = {
        'Experiment 1': [sum(inner) / len(inner) for inner in stress_exp['gd']],
        'Experiment 2': [sum(inner) / len(inner) for inner in stress_exp['preference']],        
    }


    # data = {
    #     "group": sum(([key] * len(val) for key,val in means.items()), start=[]), 
    #     "value": sum(([float(x) for x in val] for val in means.values()), start=[])
    # }


    for key,val in means.items():
        for x in val:
            data['group'].append(key)
            data['value'].append(float(x))


gd_experiment = [x for label,x in zip(data['group'], data['value']) if label == 'Experiment 1']
pr_experiment = [x for label,x in zip(data['group'], data['value']) if label == 'Experiment 2']

data = pd.DataFrame(data)
# fig = plt.figure()

# ax = sns.violinplot(data=data,x='group',y='value',inner='point')
# fig.add_axes(ax)
# ax.set_ylim(0,1)

from itertools import combinations
from scipy.stats import mannwhitneyu,wilcoxon
test = mannwhitneyu(gd_experiment, pr_experiment)
# test = wilcoxon(means[k1],means[k2])
print(f'pvalue is {test.pvalue} which is {'' if test.pvalue < 0.05 * (2/12) else 'not'} significant')
print()
# plt.show()

import seaborn as sns
import pandas as pd

plt.clf()
fig = plt.figure()
fig.set_dpi(100)

sns.stripplot(data=data,x='value', y='group',alpha=0.2)
ax = sns.pointplot(data=data, y="group", x="value", ci=95, capsize=0.3,join=False,color='black')
# sns.violinplot(data=data,y='group',x='value',inner=None,ax=ax)
ax.set_title(f"Confidence intervals for mean {var} \noverall")
ax.set_xlabel(f"{var} (s)")
ax.set_ylabel("")

ax.set_xlim(0,35)

fig.add_axes(ax)
fig.set_size_inches(4,2)
# plt.show()    
fig.savefig(f'figures/all_combined_interval_{var}.pdf',bbox_inches="tight")

print(data.head())
print(data.shape)