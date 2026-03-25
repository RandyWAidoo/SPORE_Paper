import pandas as pd
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import os


#Setup
df = pd.read_csv(sys.argv[1])["Dataset,N,D',Algorithm,Anytime,Everytime".split(',')]
max_len = df["Anytime"].apply(lambda x: len(json.loads(x))).max()
print(f"Max_Trials: {max_len}\n")

#Plot anytime performance
algo_to_anyt = {}
for algo, tbl in df.groupby("Algorithm"):
    anyt_arr = []
    for anyt_str in tbl["Anytime"].values:
        anyt = json.loads(anyt_str)
        if not len(anyt):
            anyt = [0]
        anyt = np.concat([ anyt, [anyt[-1]]*(max_len - len(anyt)) ])
        anyt_arr.append(anyt)
    
    algo_to_anyt[algo] = np.mean(anyt_arr, axis=0)

for algo, anyt in algo_to_anyt.items():
    plt.plot(list(range(1, len(anyt) + 1)), anyt, label=algo, marker='o', markersize=4, linewidth=2)
plt.xlabel("Trial")
plt.ylabel("Average Best-So-Far ARI")
plt.legend(loc="lower center", bbox_to_anchor=(0.53, -0.35), ncol=3) 
plt.savefig(os.path.join(os.path.dirname(__file__), "anytime.png"), dpi=300, bbox_inches="tight", pad_inches=0.1, )
plt.show()

#Trials-to-ARI
records = []
trials_to_ari_bounds = (0.3, 1.0)
interval = 0.05
trials_to_ari_n_values = round((trials_to_ari_bounds[1] - trials_to_ari_bounds[0])/interval) + 1
aris = np.linspace(*trials_to_ari_bounds, trials_to_ari_n_values)
print('='*25 + 'Trials to Average Anytime-ARI' + '='*25)
for algo, anyt in algo_to_anyt.items():
    trials_to_ari = []
    for ari in aris:
        trial = len(anyt) - np.sum(np.where(anyt >= ari, 1, 0)) + 1
        if trial > len(anyt):
            trial = np.inf
        trials_to_ari.append(trial)
    records.append([algo, *trials_to_ari, np.max(anyt).round(4), anyt[0].round(4)])
print(pd.DataFrame(columns=["Algorithm", *aris, "Max", "Default"], data=np.array(records)).sort_values(by="Max", ascending=False))
