import os
import yaml
import matplotlib.pyplot as plt

files = [f for f in os.listdir(
    "metrics") if os.path.isfile(os.path.join("metrics", f))]
yaml_files = [f for f in files if f.endswith(".yaml")]

current_mode = None
current_exp = -1
first = True

yaml_files.sort()

for file in yaml_files:
    numbers = yaml.safe_load(open(f'metrics/{file}', "r"))
    params = file.replace(".", "_").split("_")
    # ['flock', '08', '05', 'overall', 'avg', 'yaml']
    mode = params[0]
    exp = int(params[1])
    robots = int(params[2])

    if first is not True and (current_mode != mode or current_exp != exp):
        plt.title("Avg Pairwise Distance")
        plt.xlabel("Frame Count")
        plt.ylabel("Avg Pairwise Distance")
        plt.legend(title="Amount of Robots")
        plt.savefig(f"./metrics/{mode}_{exp}_graph.png")
        plt.close()
        print("Save", current_mode, current_exp)
        current_mode = mode
        current_exp = exp

    even = range(len(numbers.keys()))
    plt.plot(even, numbers.values(), label=robots)
    plt.xticks(even, numbers.keys())
    first = False
