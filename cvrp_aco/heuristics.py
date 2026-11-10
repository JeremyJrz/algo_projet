"""Constructive heuristic used as baseline and to initialize the pheromones:
Clarke and Wright savings (1964), parallel version."""
import numpy as np


def clarke_wright(inst):
    """Return a list of routes [0, ..., 0] built by the savings algorithm."""
    n, cap, dem, cm = inst.n, inst.capacity, inst.demand_list, inst.cost
    routes = {i: [i] for i in range(1, n + 1)}      # route id -> customers
    load = {i: dem[i] for i in routes}
    owner = {i: i for i in routes}                  # customer -> route id

    iu, ju = np.triu_indices(n, k=1)
    i_, j_ = iu + 1, ju + 1
    savings = cm[0, i_] + cm[0, j_] - cm[i_, j_]
    order = np.argsort(-savings, kind="stable")

    for t in order:
        if savings[t] <= 0:
            break
        i, j = int(i_[t]), int(j_[t])
        ri, rj = owner[i], owner[j]
        if ri == rj or load[ri] + load[rj] > cap:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] == i and b[0] == j:
            new = a + b
        elif a[0] == i and b[-1] == j:
            new = b + a
        elif a[0] == i and b[0] == j:
            new = a[::-1] + b
        elif a[-1] == i and b[-1] == j:
            new = a + b[::-1]
        else:                                       # i or j is not an endpoint
            continue
        routes[ri] = new
        load[ri] += load[rj]
        del routes[rj], load[rj]
        for c in new:
            owner[c] = ri
    return [[0] + r + [0] for r in routes.values()]
