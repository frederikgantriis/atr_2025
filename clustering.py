import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm

filenames = [f for f in os.listdir(
    "logs") if os.path.isfile(os.path.join("logs", f))]

shutil.rmtree("metrics", ignore_errors=True)
os.makedirs("metrics", exist_ok=True)

pbar = tqdm(total=len(filenames))

for filename in filenames:
    results = np.genfromtxt(os.path.join("logs", filename), delimiter=",")

    # labels = KMeans(4).fit_predict(results)
    labels = DBSCAN(eps=80, min_samples=2).fit_predict(results)

    plt.figure()
    plt.scatter(results[:, 0], results[:, 1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim([0, 800])
    plt.ylim([0, 600])
    plt.title("Positions")
    plt.grid(True)
    plt.savefig(f"./metrics/{filename}_positions.png")
    plt.close()

    # Plot clusters
    unique_labels = set(labels)
    cmap = plt.colormaps.get_cmap("tab10")
    unique_labels = sorted(set(labels) - {-1})  # exclude noise for colors

    for i, k in enumerate(unique_labels):
        xy = results[labels == k]
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            s=60,
            color=cmap(i),
            label=f"Cluster {k}"
        )

    # Optional: add noise in gray
    if -1 in labels:
        xy = results[labels == -1]
        plt.scatter(xy[:, 0], xy[:, 1], s=60,
                    color="lightgray", label="No Group")

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim([0, 800])
    plt.ylim([0, 600])
    plt.title('DBSCAN Clusters')
    plt.grid(True)
    plt.savefig(f"./metrics/{filename}_clusters.png")
    plt.close()

    # Compute average pairwise distance within each cluster
    with open("./metrics/overview.txt", "a") as f:
        f.write(f"{filename}\n")
        f.write("-------------------\n")

        for k in unique_labels:
            if k == -1:
                continue  # skip noise
            cluster_points = results[labels == k]
            if len(cluster_points) > 1:
                dists = pairwise_distances(cluster_points)
                avg_dist = np.sum(dists) / (len(cluster_points)
                                            * (len(cluster_points) - 1))

                f.write(
                    f'Cluster {k}: avg pairwise distance = {avg_dist:.2f}\n'
                )

        f.write("\n")

    pbar.update(1)

pbar.close()
