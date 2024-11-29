import numpy as np 
import json 
import seaborn as sns 
import pylab as plt
import pandas as pd


for n in [10,25,50]:
    n = str(n)
    stress_exp = list()
    for i,fname in enumerate(["nt_stress50-cleaned.json", "stress50-cleaned.json", "stress_expert-cleaned.json", "preference50-cleaned.json"]):
        with open(f"json_data/{fname}", 'r') as fdata:
            stress = json.load(fdata)

        if 'expert' not in fname:
            stress_exp.append([s['accuracy'] for s in stress])
        else: stress_exp.append([s[n]['accuracy'] for s in stress])


    stress_exp = {key: s for key,s in zip(['untrained', 'trained', 'expert', 'preference'], stress_exp)}

    means = {
        key: np.mean(np.array(tab), axis=1) for key, tab in stress_exp.items()
    }

    data = {
        "group": sum(([key] * len(val) for key,val in means.items()), start=[]), 
        "value": sum(([float(x) for x in val] for val in means.values()), start=[])
    }

    data = pd.DataFrame(data)

    fig = plt.figure()

    ax = sns.violinplot(data=data,x='group',y='value',inner='point')
    fig.add_axes(ax)
    ax.set_ylim(0,1)

    from itertools import combinations
    from scipy.stats import mannwhitneyu
    for k1,k2 in combinations(means.keys(),2):
        print(f'{k1} vs. {k2}')
        test = mannwhitneyu(means[k1], means[k2])
        print(f'pvalue is {test.pvalue}')
        print()

    fig.savefig(f"figures/n{n}_violin.pdf")
    # plt.show()

    import seaborn as sns
    import pandas as pd

    plt.clf()
    fig = plt.figure()

    # Interval plot using Seaborn's pointplot
    ax = sns.pointplot(data=data, y="group", x="value", ci=95, capsize=0.2,join=False)
    sns.stripplot(data=data,x='value', y='group',alpha=0.2)
    # sns.violinplot(data=data,y='group',x='value',inner=None,ax=ax)
    ax.set_title("Confidence intervals for mean \"accuracy\" ")
    ax.set_xlabel("accuracy")
    
    ax.set_xlim(0,1)

    fig.add_axes(ax)
    # plt.show()    
    fig.savefig(f'figures/n{n}_interval.pdf')