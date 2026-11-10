"""Lower bounds on the optimal cost of the CVRP (symmetric costs).

lb_degree : cheap bound from the degree constraints (valid for any n).
lb_lp     : linear relaxation of the two-index ILP of section 6 of the notebook,
            strengthened with rounded capacity cuts found by a separation heuristic
            (connected components of the support graph). Every cut is valid, so the
            value is a valid lower bound; it takes the capacity and the fleet limit
            into account. Intended for n up to a few hundred customers.
"""
import itertools
import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.csgraph import connected_components


def lb_degree(inst):
    """1/2 * (sum over customers of the cheapest way to leave and enter it
              + cost of the 2*m_min cheapest depot traversals)."""
    c, n = inst.cost, inst.n
    total = 0
    for i in range(1, n + 1):
        row = np.sort(np.delete(c[i], i))
        total += min(row[0] + row[1], 2 * c[i, 0])        # an edge can be used twice only at the depot
    depot = np.sort(np.repeat(c[0, 1:], 2))               # each depot edge usable up to twice
    total += depot[:2 * inst.min_routes].sum()
    return math.ceil(total / 2 - 1e-9)


def lb_lp(inst, max_rounds=150, tol=1e-6):
    """Return (lower bound, number of cuts, rounds)."""
    n, c, q, cap = inst.n, inst.cost, inst.demands, inst.capacity
    edges = [(i, j) for i in range(n + 1) for j in range(i + 1, n + 1)]
    idx = {e: t for t, e in enumerate(edges)}
    ne = len(edges)
    nv = ne + 1                                           # last variable = m (number of routes)
    cost_vec = np.array([c[i, j] for i, j in edges] + [0.0], dtype=float)
    ei = np.array([e[0] for e in edges])
    ej = np.array([e[1] for e in edges])

    # equality constraints: degree 2 at each customer; degree(depot) = 2 m
    rows, cols, vals = [], [], []
    for t, (i, j) in enumerate(edges):
        for v in (i, j):
            if v != 0:
                rows.append(v - 1); cols.append(t); vals.append(1.0)
        if i == 0:
            rows.append(n); cols.append(t); vals.append(1.0)
    rows.append(n); cols.append(ne); vals.append(-2.0)
    A_eq = coo_matrix((vals, (rows, cols)), shape=(n + 1, nv)).tocsr()
    b_eq = np.concatenate((np.full(n, 2.0), [0.0]))
    bounds = [(0, 2) if i == 0 else (0, 1) for i, _ in edges] + [(inst.min_routes, inst.num_vehicles)]

    cut_rows, cut_rhs = [], []
    seen = set()

    def add_cut(S):
        key = frozenset(S)
        if key in seen:
            return
        seen.add(key)
        inside = np.zeros(n + 1, dtype=bool)
        inside[list(S)] = True
        mask = inside[ei] != inside[ej]
        row = np.zeros(nv)
        row[:ne][mask] = -1.0                              # -x(delta(S)) <= -2 ceil(q(S)/Q)
        cut_rows.append(csr_matrix(row))
        cut_rhs.append(-2.0 * math.ceil(q[list(S)].sum() / cap))

    value = None
    for rnd in range(1, max_rounds + 1):
        res = linprog(cost_vec, A_ub=vstack(cut_rows) if cut_rows else None,
                      b_ub=np.array(cut_rhs) if cut_rows else None,
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.status != 0:
            break
        value, x = res.fun, res.x[:ne]
        added = 0
        for thr in (tol, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9):   # components of the support graph
            mask = (x > thr) & (ei > 0)                    # customer-customer edges only
            g = coo_matrix((np.ones(mask.sum()), (ei[mask] - 1, ej[mask] - 1)), shape=(n, n))
            ncomp, label = connected_components(g, directed=False)
            for comp in range(ncomp):
                S = [int(v) + 1 for v in np.flatnonzero(label == comp)]
                inside = np.zeros(n + 1, dtype=bool)
                inside[S] = True
                crossing = x[inside[ei] != inside[ej]].sum()
                need = 2 * math.ceil(q[S].sum() / cap)
                if crossing < need - 1e-6 and frozenset(S) not in seen:
                    add_cut(S)
                    added += 1
        if added == 0:
            break
    if value is None:
        return None, len(cut_rows), rnd
    return math.ceil(value - 1e-6), len(cut_rows), rnd
