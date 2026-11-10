"""Figures: routes, convergence, quality distribution, statistics by factor."""
import matplotlib.pyplot as plt
import numpy as np


def plot_solution(inst, routes, title="Routes", show_demands=None, ax=None):
    """Routes on the plane (needs coordinates). Demands are written when n is small."""
    if inst.coords is None:
        raise ValueError("this instance has no coordinates")
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 6))
    show_demands = inst.n <= 40 if show_demands is None else show_demands
    for k, r in enumerate(routes):
        pts = inst.coords[r]
        ax.plot(pts[:, 0], pts[:, 1], "-o", lw=1.8, ms=4, label=f"route {k + 1}")
    ax.scatter(*inst.coords[0], marker="s", s=150, c="gold", edgecolors="black", zorder=5, label="depot")
    if show_demands:
        for i in range(1, inst.n + 1):
            ax.annotate(f"{inst.demand_list[i]}", inst.coords[i], textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    if len(routes) <= 12:
        ax.legend(fontsize=7)
    return ax


def plot_convergence(results, labels=None, ax=None, title="Convergence"):
    """Best-so-far fitness per iteration for one or several runs."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 4))
    results = results if isinstance(results, (list, tuple)) else [results]
    for k, r in enumerate(results):
        ax.plot(range(1, len(r.history) + 1), r.history,
                label=None if labels is None else labels[k])
    ax.set_xlabel("iteration")
    ax.set_ylabel("best cost so far")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if labels:
        ax.legend()
    return ax


def plot_ant_distribution(result, ax=None, title="Quality of the ants over the iterations"):
    """Box plot of the cost of the ants for a few iterations (quality distribution)."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 4))
    its = np.unique(np.linspace(0, len(result.ant_costs) - 1, 8).astype(int))
    ax.boxplot([result.ant_costs[i] for i in its])
    ax.set_xticklabels([str(i + 1) for i in its])
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost of the ants")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def boxplot_by(df, by, metric, ax=None, title=None, order=None):
    """Box plot of a metric grouped by a factor of the experimental design."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 4))
    groups = order if order is not None else sorted(df[by].unique())
    ax.boxplot([df.loc[df[by] == g, metric].dropna() for g in groups])
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_xlabel(by)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by {by}")
    ax.grid(alpha=0.3)
    return ax


def plot_time_vs_n(df, ax=None, title="CPU time versus number of customers"):
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 4))
    g = df.groupby("n")["cpu_time"].agg(["median"]).reset_index()
    ax.loglog(g["n"], g["median"], "o-")
    ax.set_xlabel("n (customers)")
    ax.set_ylabel("median CPU time (s)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    return ax
