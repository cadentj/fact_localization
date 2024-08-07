# %%

import matplotlib.pyplot as plt 
import numpy as np
import json


for layer in [6,17,38]:

    path = f"results/{layer}.json"

    points = {"x" : [], "y" : []}

    with open(path, "r") as f:
        results = json.load(f)

    with open("results/tracing_effect.json", "r") as f:
        tracing = json.load(f)

    for result, tracing in zip(results.values(), tracing):

        rewrite_score = (result['new'] - result['init']) / (1 - result['init'])

        if tracing['idxs'][layer] == tracing['subject_idxs'][-1]:
            tracing_score = tracing['vals'][layer]

            points["x"].append(tracing_score)
            points["y"].append(rewrite_score)

        else:
            continue

    x = np.array(points["x"])
    y = np.array(points["y"])

    plt.scatter(x, y, alpha=0.5)
    plt.title(f"Layer {layer}")
    plt.xlabel("Tracing Score")
    plt.ylabel("Rewrite Score")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
