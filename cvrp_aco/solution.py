"""Basic operations on solutions.

A solution is a list of routes; a route is a list of vertices [0, i1, ..., im, 0]
(vertex 0 is the depot). Costs are read in a list of lists (fast in pure Python).
"""
import numpy as np


def route_cost(route, cost):
    return sum(cost[route[i]][route[i + 1]] for i in range(len(route) - 1))


def solution_cost(routes, cost):
    return sum(route_cost(r, cost) for r in routes)


def route_load(route, demands):
    return sum(demands[i] for i in route)


def is_feasible(routes, inst, check_fleet=True):
    """Each customer once, routes start/end at 0, capacity respected, at most k routes."""
    if check_fleet and len(routes) > inst.num_vehicles:
        return False
    seen = []
    for r in routes:
        if len(r) < 3 or r[0] != 0 or r[-1] != 0:
            return False
        inner = r[1:-1]
        if 0 in inner or route_load(r, inst.demand_list) > inst.capacity:
            return False
        seen.extend(inner)
    return sorted(seen) == list(range(1, inst.n + 1))


def edge_arrays(routes):
    """Arrays (a, b) of all the edges travelled by a solution (used to update pheromones)."""
    a, b = [], []
    for r in routes:
        a.extend(r[:-1])
        b.extend(r[1:])
    return np.array(a, dtype=int), np.array(b, dtype=int)
