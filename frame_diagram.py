import os
from collections import defaultdict

import matplotlib.pyplot as plt
import yaml
import numpy as np
from tqdm import trange

mode = "disperse"

files = [f for f in os.listdir(
    "metrics") if os.path.isfile(os.path.join("metrics", f))]
yaml_files = [f for f in files if f.endswith(".yaml")]
yaml_files = [f for f in yaml_files if mode in f]

current_mode = None
current_exp = -1
first = True

yaml_files.sort()

avg = 0
robots = defaultdict(dict)

for file in yaml_files:
    numbers = yaml.safe_load(open(f'metrics/{file}', "r"))
    params = file.replace(".", "_").split("_")
    # ['flock', '08', '05', 'overall', 'avg', 'yaml']
    mode = params[0]
    exp = int(params[1])
    key = int(params[2])
    for k, v in numbers.items():
        if k not in robots[key].keys():
            robots[key][k] = [v]
        else:
            robots[key][k] += [v]

overall_avg = defaultdict(dict)

for rkey, rval in robots.items():
    for fkey, fval in rval.items():
        overall_avg[rkey][fkey] = np.average(fval)


pbar = trange(len(overall_avg.keys()))

for k, v in overall_avg.items():
    pbar.set_description(f"Working on {k}")
    even = range(len(overall_avg[k].keys()))
    plt.plot(even, overall_avg[k].values(), label=k)
    plt.xticks(even, overall_avg[k].keys())
    pbar.update()

plt.title("Avg Pairwise Distance")
plt.xlabel("Frame Count")
plt.ylabel("Avg Pairwise Distance")
plt.legend(title="Amount of Robots")
plt.savefig(f"./metrics/{mode}_graph.png")
plt.close()
