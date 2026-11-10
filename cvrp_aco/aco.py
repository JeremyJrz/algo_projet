"""Ant colony optimization for the CVRP (two variants, optional local search).

variant = "acs": Ant Colony System (Dorigo and Gambardella, 1997)
    - pseudo-random proportional rule (greedy choice with probability q0),
    - local pheromone update on every edge crossed by an ant,
    - global update: only the best-so-far solution evaporates / deposits,
    - candidate lists, Clarke and Wright initialization.
variant = "as": Elitist Ant System (first prototype of the team)
    - roulette-wheel rule only,
    - every ant deposits Q / L after global evaporation, best-so-far is reinforced.

Both variants build the routes one after the other: a route is closed when no
unvisited customer fits in the remaining capacity. If more than k routes are needed,
the solution is penalized (penalty_factor * max cost per extra route).
"""
import time
from dataclasses import dataclass, field

import numpy as np

from configs import ACOConfig
from heuristics import clarke_wright
from local_search import local_search
from solution import edge_arrays, is_feasible, solution_cost


@dataclass
class ACOResult:
    routes: list
    cost: float                    # real cost of the best feasible-or-penalized solution
    fitness: float                 # cost + fleet penalty
    feasible: bool
    history: list                  # best-so-far fitness after each iteration
    iteration_best: list           # best fitness of each iteration
    ant_costs: list                # per iteration: fitness of every ant
    iterations: int
    iter_of_best: int              # iteration where the final best was found
    cpu_time: float
    config: ACOConfig = field(repr=False, default=None)


def _construct(inst, cfg, tau, eta_beta, cand, rng, tau0):
    """One ant builds a complete solution (list of routes)."""
    n, cap, q = inst.n, inst.capacity, inst.demands
    unvisited = np.ones(n + 1, dtype=bool)
    unvisited[0] = False
    remaining = n
    routes = []
    acs = cfg.variant == "acs"
    while remaining > 0:
        route, cur, load = [0], 0, 0
        while True:
            pool = cand[cur] if cand is not None else None
            if pool is not None:
                ok = pool[unvisited[pool] & (load + q[pool] <= cap)]
            else:
                ok = np.empty(0, dtype=int)
            if len(ok) == 0:                                  # fall back on all customers
                ok = np.flatnonzero(unvisited & (load + q <= cap))
            if len(ok) == 0:
                break                                          # nothing fits: close the route
            w = tau[cur, ok] ** cfg.alpha * eta_beta[cur, ok]
            if acs and rng.random() < cfg.q0:
                nxt = int(ok[np.argmax(w)])                    # exploitation
            else:                                              # biased exploration (roulette)
                total = w.sum()
                if not np.isfinite(total) or total <= 0:
                    nxt = int(ok[rng.integers(len(ok))])
                else:
                    nxt = int(ok[min(np.searchsorted(np.cumsum(w), rng.random() * total), len(ok) - 1)])
            if acs:                                            # local pheromone update
                v = (1 - cfg.xi) * tau[cur, nxt] + cfg.xi * tau0
                tau[cur, nxt] = tau[nxt, cur] = v
            route.append(nxt)
            load += int(q[nxt])
            unvisited[nxt] = False
            remaining -= 1
            cur = nxt
        route.append(0)
        routes.append(route)
    return routes


def run_aco(inst, cfg: ACOConfig, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    t_start = time.process_time()
    cost_list = inst.cost_list
    size = inst.n + 1
    eta = 1.0 / (inst.cost.astype(float) + 1e-6)
    np.fill_diagonal(eta, 0.0)
    eta_beta = eta ** cfg.beta
    cand = inst.candidates(cfg.candidate_size) if cfg.candidate_size is not None else None
    penalty = cfg.penalty_factor * float(inst.cost.max())

    def fitness_of(routes):
        return solution_cost(routes, cost_list) + penalty * max(0, len(routes) - inst.num_vehicles)

    best_routes, best_fit = None, float("inf")
    if cfg.init_with_greedy or cfg.variant == "acs":
        greedy = clarke_wright(inst)
        if cfg.ls_scope != "none":                       # hybrid: start from a locally optimal solution
            greedy = local_search(greedy, inst, cfg.ls_neighbors)
        if cfg.init_with_greedy:
            best_routes, best_fit = greedy, fitness_of(greedy)
        base = solution_cost(greedy, cost_list)
    if cfg.variant == "acs":
        tau0 = 1.0 / (inst.n * base)
    else:
        tau0 = cfg.tau_init
    tau = np.full((size, size), tau0, dtype=float)

    history, iter_best, ant_costs = [], [], []
    iter_of_best, stale = 0, 0
    it = 0
    for it in range(1, cfg.max_iterations + 1):
        ants = []
        for _ in range(cfg.num_ants):
            routes = _construct(inst, cfg, tau, eta_beta, cand, rng, tau0)
            if cfg.ls_scope == "all":
                routes = local_search(routes, inst, cfg.ls_neighbors)
            ants.append((fitness_of(routes), routes))
        if cfg.ls_scope == "iteration_best":
            k = min(range(len(ants)), key=lambda t: ants[t][0])
            improved = local_search(ants[k][1], inst, cfg.ls_neighbors)
            ants[k] = (fitness_of(improved), improved)
        ant_costs.append([a[0] for a in ants])
        fit_it, routes_it = min(ants, key=lambda a: a[0])
        iter_best.append(fit_it)

        if fit_it < best_fit - 1e-9:
            best_fit, best_routes = fit_it, [r[:] for r in routes_it]
            iter_of_best, stale = it, 0
        else:
            stale += 1

        # ---- global pheromone update ------------------------------------------------
        if cfg.variant == "acs":
            a, b = edge_arrays(best_routes)
            val = (1 - cfg.rho) * tau[a, b] + cfg.rho / best_fit
            tau[a, b] = val
            tau[b, a] = val
        else:
            tau *= (1 - cfg.rho)
            for fit, routes in ants:
                a, b = edge_arrays(routes)
                np.add.at(tau, (a, b), cfg.deposit / fit)
                np.add.at(tau, (b, a), cfg.deposit / fit)
            a, b = edge_arrays(best_routes)
            extra = cfg.elitist_weight * cfg.deposit / best_fit
            np.add.at(tau, (a, b), extra)
            np.add.at(tau, (b, a), extra)
            np.clip(tau, cfg.tau_min, cfg.tau_max, out=tau)
        history.append(best_fit)

        if verbose and it % 10 == 0:
            print(f"  iteration {it:4d} | best = {best_fit:.0f}")
        if cfg.patience is not None and stale >= cfg.patience:
            break
        if cfg.time_limit is not None and time.process_time() - t_start >= cfg.time_limit:
            break

    if cfg.ls_scope != "none":                                    # final polish of the best
        polished = local_search(best_routes, inst, cfg.ls_neighbors)
        if fitness_of(polished) < best_fit - 1e-9:
            best_routes, best_fit = polished, fitness_of(polished)
            iter_of_best = it
            history[-1] = best_fit

    cost = solution_cost(best_routes, cost_list)
    return ACOResult(best_routes, cost, best_fit, is_feasible(best_routes, inst), history,
                     iter_best, ant_costs, it, iter_of_best,
                     time.process_time() - t_start, cfg)
