"""Experimental design: instance suite, runs over seeds, parameter sweeps, statistics."""
import numpy as np
import pandas as pd

from aco import run_aco
from bounds import lb_degree, lb_lp
from heuristics import clarke_wright
from local_search import local_search
from instances import generate_instance
from solution import solution_cost


def default_suite(quick=False, large=False):
    """The instance scenarios (21 instances; each family changes ONE factor).

    A  size                : n = 20 ... 200 (random layout, tightness 0.9, complete graph)
    B  customer layout     : clustered instead of random
    C  capacity tightness  : 0.7, 0.8, 0.95, 0.99 (n = 100)
    D  graph density       : sparse graphs (5 %, 10 %, 25 % of the edges), then completed
    E  depot position      : center, corner
    F  capacity (vehicle size): small and large vehicles
    """
    S = []
    base = dict(layout="random", tightness=0.9, capacity=30, density=1.0, depot="random")

    def add(tag, n, **kw):
        spec = {**base, **kw, "n": n, "name": f"{tag}-n{n}"}
        S.append(spec)

    for n in (20, 30, 50, 75, 100, 150, 200):
        add("A-size", n)
    for n in (75, 100, 150):
        add("B-clustered", n, layout="clustered")
    for t in (0.7, 0.8, 0.95, 0.99):
        add(f"C-tight{t}", 100, tightness=t)
    for d in (0.05, 0.10, 0.25):
        add(f"D-dens{d}", 100, density=d)
    for dep in ("center", "corner"):
        add(f"E-{dep}", 100, depot=dep)
    add("F-smallQ", 100, capacity=15)
    add("F-largeQ", 100, capacity=60)
    if large:
        for n in (500, 1000):
            add("G-large", n)
    if quick:
        S = [s for s in S if s["n"] <= 50 or s["name"].startswith(("D-", "E-"))][:6]
    return S


def build_instances(suite, seed=2026):
    out = []
    for k, spec in enumerate(suite):
        spec = dict(spec)
        name = spec.pop("name")
        n = spec.pop("n")
        out.append(generate_instance(name, n, seed=seed + k, grid=max(100, 4 * n), **spec))
    return out


def run_suite(instances, cfg, seeds, bounds=True, lp_max_n=100, label=None, verbose=True):
    """Run the algorithm on every instance with several seeds; return a DataFrame."""
    rows = []
    for inst in instances:
        cw_routes = clarke_wright(inst)
        cw = solution_cost(cw_routes, inst.cost_list)
        cw_ls = solution_cost(local_search(cw_routes, inst, cfg.ls_neighbors), inst.cost_list)
        lb = None
        if bounds:
            lb = lb_lp(inst)[0] if inst.n <= lp_max_n else lb_degree(inst)
        if verbose:
            print(f"{inst.summary()} | CW = {cw} | LB = {lb}")
        for s in seeds:
            r = run_aco(inst, cfg, seed=s)
            rows.append(dict(
                instance=inst.name, n=inst.n, k=inst.num_vehicles, capacity=inst.capacity,
                layout=inst.meta.get("layout"), depot=inst.meta.get("depot"),
                tightness=round(inst.tightness, 2), density=round(inst.density, 2),
                mean_degree=round(inst.mean_degree, 1), seed=s, variant=label or cfg.variant,
                cost=r.cost, feasible=r.feasible, n_routes=len(r.routes),
                cpu_time=r.cpu_time, iterations=r.iterations, iter_of_best=r.iter_of_best,
                cw_cost=cw, cw_ls_cost=cw_ls, lower_bound=lb,
                gap_lb=None if lb is None else 100 * (r.cost - lb) / lb,
                gain_vs_cw=100 * (cw - r.cost) / cw,
                gain_vs_cw_ls=100 * (cw_ls - r.cost) / cw_ls))
    return pd.DataFrame(rows)


def sweep(inst_list, cfg, param, values, seeds, label=None):
    """Vary one algorithm parameter and keep the other ones fixed."""
    frames = []
    for v in values:
        df = run_suite(inst_list, cfg.but(**{param: v}), seeds, bounds=False, label=label, verbose=False)
        df[param] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def mad(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x))))


def summarize(df, by, metrics=("gap_lb", "gain_vs_cw", "gain_vs_cw_ls", "cpu_time", "iterations", "iter_of_best")):
    """Mean, standard deviation, median and median absolute deviation by factor."""
    agg = {}
    for m in metrics:
        if m in df:
            agg[m] = ["count", "mean", "std", "median", mad]
    out = df.groupby(by).agg(agg)
    out.columns = [f"{m}_{'mad' if s == 'mad' else s}" for m, s in out.columns]
    return out.round(2)
