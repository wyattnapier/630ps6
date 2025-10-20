import random
from heapdict import heapdict
import matplotlib.pyplot as plt
import numpy as np
from math import comb

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

def run_experiment():
    ns = [500, 5000, 50000]
    num_runs = 20
    # random.seed(42)

    results = {}

    for n in ns:
        print(f"Running experiments for n={n}...")
        runs_min = []
        runs_max = []
        tbig_vals, tconn_vals, tnoiso_vals, diff_vals = [], [], [], []

        for _ in range(num_runs):
            minComp, maxComp = RandomConnections(n)
            runs_min.append(minComp)
            runs_max.append(maxComp)

            tbig = next((i for i, val in enumerate(maxComp) if val >= n / 2), None)
            tconn = next((i for i, val in enumerate(maxComp) if val == n), None)
            tnoiso = next((i for i, val in enumerate(minComp) if val > 1), None)

            tbig_vals.append(tbig if tbig is not None else np.nan)
            tconn_vals.append(tconn if tconn is not None else np.nan)
            tnoiso_vals.append(tnoiso if tnoiso is not None else np.nan)
            diff_vals.append((tconn - tnoiso) if (tconn and tnoiso) else np.nan)

        # Pad runs so we can average across different lengths
        max_len = max(len(r) for r in runs_max)
        min_avg = np.zeros(max_len)
        max_avg = np.zeros(max_len)
        counts = np.zeros(max_len)

        for minC, maxC in zip(runs_min, runs_max):
            for i in range(len(maxC)):
                min_avg[i] += minC[i]
                max_avg[i] += maxC[i]
                counts[i] += 1

        min_avg /= np.where(counts == 0, 1, counts)
        max_avg /= np.where(counts == 0, 1, counts)

        results[n] = {
            "min_avg": min_avg,
            "max_avg": max_avg,
            "tbig": tbig_vals,
            "tconn": tconn_vals,
            "tnoiso": tnoiso_vals,
            "diff": diff_vals,
        }

        # Plot average maxComp and minComp over time
        plt.figure(figsize=(10, 6))
        plt.plot(max_avg, label='avg maxComp_t')
        plt.plot(min_avg, label='avg minComp_t')
        plt.title(f"Average Component Sizes over Time (n={n})")
        plt.xlabel("t (edges added)")
        plt.ylabel("Component size")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot histograms of tbig, tconnect, tnoiso
        plt.figure(figsize=(10, 6))
        plt.hist([x for x in tbig_vals if not np.isnan(x)], bins=15, alpha=0.6, label="tbig")
        plt.hist([x for x in tconn_vals if not np.isnan(x)], bins=15, alpha=0.6, label="tconnect")
        plt.hist([x for x in tnoiso_vals if not np.isnan(x)], bins=15, alpha=0.6, label="tno-iso")
        plt.title(f"Histograms of Key Times (n={n})")
        plt.xlabel("t")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot histogram of (tconnect - tnoiso)
        plt.figure(figsize=(8, 5))
        plt.hist([x for x in diff_vals if not np.isnan(x)], bins=15, color='purple', alpha=0.7)
        plt.title(f"Histogram of (tconnect - tno-iso) (n={n})")
        plt.xlabel("tconnect - tno-iso")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results

if __name__ == "__main__":
    run_experiment()
