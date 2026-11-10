"""Local search for the CVRP: 2-opt (inside a route), relocate and swap (between routes).

Moves are evaluated with an incremental cost difference (delta) and accepted only if
they improve the cost and keep the capacity satisfied (first-improvement strategy).
Relocate and swap only look at the nearest neighbors of each customer.
"""


def _two_opt(route, c):
    """2-opt inside one route [0, ..., 0]; returns True if the route changed."""
    changed = False
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                a, b, x, y = route[i - 1], route[i], route[j], route[j + 1]
                if c[a][x] + c[b][y] < c[a][b] + c[x][y]:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    improved = changed = True
    return changed


def _positions(routes):
    return {v: (r, p) for r, route in enumerate(routes) for p, v in enumerate(route) if v != 0}


def _relocate(routes, loads, inst, near):
    c, q, cap = inst.cost_list, inst.demand_list, inst.capacity
    pos = _positions(routes)
    for u, (ra, pa) in list(pos.items()):
        A = routes[ra]
        prev, nxt = A[pa - 1], A[pa + 1]
        gain = c[prev][u] + c[u][nxt] - c[prev][nxt]          # saving when u is removed
        for v in near[u]:
            rb, pb = pos[int(v)]
            if rb == ra:
                continue
            if loads[rb] + q[u] > cap:
                continue
            B = routes[rb]
            for x, y, where in ((B[pb], B[pb + 1], pb + 1), (B[pb - 1], B[pb], pb)):
                if c[x][u] + c[u][y] - c[x][y] < gain:        # strict improvement
                    A.pop(pa)
                    routes[rb].insert(where, u)
                    loads[ra] -= q[u]
                    loads[rb] += q[u]
                    return True
    return False


def _swap(routes, loads, inst, near):
    c, q, cap = inst.cost_list, inst.demand_list, inst.capacity
    pos = _positions(routes)
    for u, (ra, pa) in pos.items():
        A = routes[ra]
        for v in near[u]:
            rb, pb = pos[int(v)]
            if rb == ra:
                continue
            if loads[ra] - q[u] + q[v] > cap or loads[rb] - q[v] + q[u] > cap:
                continue
            B = routes[rb]
            a1, a2, b1, b2 = A[pa - 1], A[pa + 1], B[pb - 1], B[pb + 1]
            old = c[a1][u] + c[u][a2] + c[b1][v] + c[v][b2]
            new = c[a1][v] + c[v][a2] + c[b1][u] + c[u][b2]
            if new < old:
                A[pa], B[pb] = v, u
                loads[ra] += q[v] - q[u]
                loads[rb] += q[u] - q[v]
                return True
    return False


def local_search(routes, inst, ls_neighbors=10, max_rounds=500):
    """Improve a solution in place (a new list of routes is returned)."""
    routes = [r[:] for r in routes]
    q = inst.demand_list
    loads = [sum(q[i] for i in r) for r in routes]
    near = inst.candidates(ls_neighbors)
    for _ in range(max_rounds):
        moved = False
        for r in routes:
            moved |= _two_opt(r, inst.cost_list)
        moved |= _relocate(routes, loads, inst, near)
        moved |= _swap(routes, loads, inst, near)
        # remove empty routes created by relocate moves
        keep = [i for i, r in enumerate(routes) if len(r) > 2]
        if len(keep) != len(routes):
            routes = [routes[i] for i in keep]
            loads = [loads[i] for i in keep]
        if not moved:
            break
    return routes
