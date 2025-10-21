import random
from heapdict import heapdict
import matplotlib.pyplot as plt
import numpy as np
from math import comb
import os

def RandomConnections(n):
    # initialize node set, graph, and structures
    V = list(range(1, n + 1))
    G = {v: [] for v in V}
    E = set()
    t = 0
    endTime = comb(n, 2) // 2  # stop condition

    parent = {}
    minHeapDict = heapdict()  # root -> component size
    minComp = [1]
    maxComp = [1]

    # initialize union-find and heapdict
    for v in V:
        parent[v] = v
        minHeapDict[v] = 1

    # --- helper functions ---
    def Find(u):
        if parent[u] != u:
            parent[u] = Find(parent[u])
        return parent[u]

    def Merge(r1, r2):
        # merge smaller component into larger one
        if minHeapDict[r1] > minHeapDict[r2]:
            r1, r2 = r2, r1
        parent[r1] = r2
        minHeapDict[r2] += minHeapDict[r1]
        del minHeapDict[r1]
        return minHeapDict[r2]

    def getRandomNewEdge(G, E):
        while True:
            u = random.randint(1, n)
            v = random.randint(1, n)
            if u == v:
                continue
            edge = tuple(sorted((u, v)))
            if edge not in E:
                E.add(edge)
                G[u].append(v)
                G[v].append(u)
                return u, v

    # --- main loop ---
    while t < endTime and maxComp[-1] != n:
        u, v = getRandomNewEdge(G, E)
        uRoot = Find(u)
        vRoot = Find(v)

        if uRoot == vRoot:
            # already connected
            minComp.append(minComp[-1])
            maxComp.append(maxComp[-1])
            t += 1
            continue

        newParentSize = Merge(uRoot, vRoot)
        maxComp.append(max(newParentSize, maxComp[-1]))
        # smallest component size currently in the heapdict
        _, minSize = minHeapDict.peekitem()
        minComp.append(minSize)
        t += 1

    return minComp, maxComp

def run_experiments():
    ns = [500, 5000, 50000]
    num_runs = 20
    random.seed(42)
    results = {}

    os.makedirs("plots", exist_ok=True)  # save all plots in ./plots/

    for n in ns:
        print(f"\nRunning experiments for n = {n}...")
        runs_min = []
        runs_max = []
        tbig_vals, tconn_vals, tnoiso_vals = [], [], []

        for _ in range(num_runs):
            minComp, maxComp = RandomConnections(n)
            runs_min.append(minComp)
            runs_max.append(maxComp)

            tbig = next((i for i, val in enumerate(maxComp) if val >= n / 2), None)
            tconn = next((i for i, val in enumerate(maxComp) if val == n), None)
            tnoiso = next((i for i, val in enumerate(minComp) if val > 1), None)

            tbig_vals.append(tbig / n if tbig is not None else np.nan)
            tconn_vals.append(tconn / n if tconn is not None else np.nan)
            tnoiso_vals.append(tnoiso / n if tnoiso is not None else np.nan)

        # ---- compute average component sizes correctly ----
        # max_len = max(len(run) for run in runs_max)
        # min_avg = np.zeros(max_len)
        # max_avg = np.zeros(max_len)
        # counts = np.zeros(max_len)

        # for minC, maxC in zip(runs_min, runs_max):
        #     for i in range(len(maxC)):
        #         min_avg[i] += minC[i]
        #         max_avg[i] += maxC[i]
        #         counts[i] += 1

        # # divide only where counts > 0, leave NaN elsewhere to avoid plotting zeros
        # min_avg = np.divide(min_avg, counts, out=np.full_like(min_avg, np.nan), where=counts>0)
        # max_avg = np.divide(max_avg, counts, out=np.full_like(max_avg, np.nan), where=counts>0)
        max_len = max(len(run) for run in runs_max)
        min_avg = np.zeros(max_len)
        max_avg = np.zeros(max_len)
        counts = np.zeros(max_len)

        for minC, maxC in zip(runs_min, runs_max):
            last_min, last_max = 1, 1  # initial values
            for i in range(max_len):
                if i < len(maxC):
                    last_min = minC[i]
                    last_max = maxC[i]
                min_avg[i] += last_min
                max_avg[i] += last_max
                counts[i] += 1

        # now divide to get average
        min_avg = min_avg / counts
        max_avg = max_avg / counts

        results[n] = {
            "min_avg": min_avg,
            "max_avg": max_avg,
            "tbig": tbig_vals,
            "tconn": tconn_vals,
            "tnoiso": tnoiso_vals
        }

        # ---- Plot 1: average component sizes ----
        plt.figure(figsize=(9, 6))
        plt.plot(max_avg, label='avg maxComp_t')
        plt.plot(min_avg, label='avg minComp_t')
        plt.title(f"Average Component Sizes vs t (n={n})")
        plt.xlabel("t (edges added)")
        plt.ylabel("Component size")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"plots/avg_components_n{n}.png", dpi=300)
        plt.close()

        # ---- Plot 2: histograms side by side ----
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Normalized Time Distributions (n={n})")

        bins = 15
        axes[0].hist([x for x in tbig_vals if not np.isnan(x)], bins=bins, color='C0', alpha=0.7)
        axes[0].set_title(r"$t_{big}/n$")
        axes[0].set_xlabel("t / n")
        axes[0].set_ylabel("Frequency")

        axes[1].hist([x for x in tconn_vals if not np.isnan(x)], bins=bins, color='C1', alpha=0.7)
        axes[1].set_title(r"$t_{connect}/n$")
        axes[1].set_xlabel("t / n")
        axes[1].set_ylabel("Frequency")

        axes[2].hist([x for x in tnoiso_vals if not np.isnan(x)], bins=bins, color='C2', alpha=0.7)
        axes[2].set_title(r"$t_{no-iso}/n$")
        axes[2].set_xlabel("t / n")
        axes[2].set_ylabel("Frequency")

        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"plots/histograms_n{n}.png", dpi=300)
        plt.close()

    return results


if __name__ == "__main__":
    run_experiments()
