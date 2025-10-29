from typing import DefaultDict
import yaml

experiments = []

for file_name in ["test_result.yaml", "test_results_2.yaml"]:
    with open(file_name, "r") as file:
        data = yaml.safe_load(file)

    dictionary = DefaultDict(int)

    for i in range(1, 4):
        experiment = data.get(f"Experiment_{i}")

        for key in experiment.keys():
            dictionary[key] += experiment[key]

    experiments.append(dictionary)

diff_dict = DefaultDict(int)

keys = experiments[0].keys()

for key in keys:
    diff_dict[key] = experiments[0][key] - experiments[1][key]

with open("diff_results.yaml", "w") as file:
    yaml.dump(diff_dict, file)
