"""Command line entry point.

Examples
--------
python main.py demo                         # one instance, ACS + local search, figures
python main.py demo --preset as             # same with the elitist Ant System
python main.py experiments --quick          # small experimental campaign (a few minutes)
python main.py experiments --seeds 10       # full campaign (21 instances x 10 seeds)
python main.py sweep --param rho --values 0.05 0.1 0.3 0.5
python main.py vrplib path/to/X-n101-k25.vrp --best-known path/to/X-n101-k25.sol
"""
import argparse
import os

import matplotlib
import pandas as pd

from configs import PRESET_ACS_HYBRID, PRESET_AS_ELITIST

PRESETS = {"acs": PRESET_ACS_HYBRID, "as": PRESET_AS_ELITIST}


def parse():
    p = argparse.ArgumentParser(description="ACO for the CVRP")
    p.add_argument("command", choices=["demo", "experiments", "sweep", "vrplib"])
    p.add_argument("path", nargs="?", help="VRPLIB .vrp file (vrplib command)")
    p.add_argument("--preset", choices=list(PRESETS), default="acs")
    p.add_argument("--ls", choices=["none", "iteration_best", "all"], help="override local-search scope")
    p.add_argument("--n", type=int, default=50, help="customers (demo)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--large", action="store_true", help="add n = 500 and 1000 to the suite")
    p.add_argument("--param", default="rho")
    p.add_argument("--values", nargs="+", type=float, default=[0.05, 0.1, 0.3, 0.5])
    p.add_argument("--best-known", help="VRPLIB .sol file")
    p.add_argument("--out", default="outputs")
    p.add_argument("--show", action="store_true", help="open the figures (otherwise they are saved)")
    return p.parse_args()


def finish(fig_name, out, show):
    import matplotlib.pyplot as plt
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(out, fig_name), dpi=130)
    plt.close()


def main():
    a = parse()
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import experiments as ex
    import plots
    from aco import run_aco
    from heuristics import clarke_wright
    from instances import generate_instance
    from solution import solution_cost

    os.makedirs(a.out, exist_ok=True)
    cfg = PRESETS[a.preset]
    if a.ls:
        cfg = cfg.but(ls_scope=a.ls)

    if a.command == "demo":
        inst = generate_instance("demo", a.n, tightness=0.9, capacity=30, seed=a.seed, grid=max(100, 4 * a.n))
        print(inst.summary())
        cw = clarke_wright(inst)
        print("Clarke and Wright:", solution_cost(cw, inst.cost_list))
        r = run_aco(inst, cfg, seed=a.seed, verbose=True)
        print(f"ACO ({cfg.variant}, ls={cfg.ls_scope}): cost {r.cost:.0f}, feasible {r.feasible}, "
              f"{len(r.routes)} routes, {r.iterations} iterations, CPU {r.cpu_time:.2f}s")
        plots.plot_solution(inst, cw, f"Clarke and Wright: {solution_cost(cw, inst.cost_list)}")
        finish("demo_clarke_wright.png", a.out, a.show)
        plots.plot_solution(inst, r.routes, f"ACO ({cfg.variant}): {r.cost:.0f}")
        finish("demo_routes.png", a.out, a.show)
        plots.plot_convergence(r)
        finish("demo_convergence.png", a.out, a.show)
        plots.plot_ant_distribution(r)
        finish("demo_ant_distribution.png", a.out, a.show)

    elif a.command == "experiments":
        suite = ex.default_suite(quick=a.quick, large=a.large)
        instances = ex.build_instances(suite)
        cfg_run = cfg.but(max_iterations=40, patience=15) if a.quick else cfg
        df = ex.run_suite(instances, cfg_run, seeds=range(a.seeds), label=a.preset)
        df.to_csv(os.path.join(a.out, f"results_{a.preset}.csv"), index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", None)
        print("\nBy instance:\n", ex.summarize(df, "instance"))
        print("\nBy size n:\n", ex.summarize(df, "n"))
        plots.boxplot_by(df, "instance", "gap_lb", title="Gap to the lower bound (%)")
        plt.xticks(rotation=60, fontsize=6)
        finish("exp_gap_by_instance.png", a.out, a.show)
        plots.plot_time_vs_n(df)
        finish("exp_time_vs_n.png", a.out, a.show)

    elif a.command == "sweep":
        suite = [s for s in ex.default_suite() if s["name"] in ("A-size-n50", "B-clustered-n75")]
        instances = ex.build_instances(suite)
        cfg_run = cfg.but(max_iterations=40, patience=15) if a.quick else cfg
        values = [int(v) if a.param in ("num_ants", "candidate_size", "ls_neighbors") else v for v in a.values]
        df = ex.sweep(instances, cfg_run, a.param, values, seeds=range(a.seeds), label=a.preset)
        df.to_csv(os.path.join(a.out, f"sweep_{a.param}_{a.preset}.csv"), index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", None)
        print(ex.summarize(df, [a.param], metrics=("gain_vs_cw", "gain_vs_cw_ls", "cpu_time", "iterations", "iter_of_best")))
        plots.boxplot_by(df, a.param, "cost", title=f"Cost versus {a.param}")
        finish(f"sweep_{a.param}.png", a.out, a.show)

    elif a.command == "vrplib":
        from instances import best_known_cost, load_vrplib
        inst = load_vrplib(a.path)
        print(inst.summary())
        r = run_aco(inst, cfg, seed=a.seed, verbose=True)
        print(f"cost {r.cost:.0f} | feasible {r.feasible} | CPU {r.cpu_time:.1f}s")
        if a.best_known:
            bk = best_known_cost(a.best_known)
            print(f"best known {bk:.0f} | gap = {100 * (r.cost - bk) / bk:.2f} %")
        if inst.coords is not None:
            plots.plot_solution(inst, r.routes, f"{inst.name}: {r.cost:.0f}")
            finish("vrplib_routes.png", a.out, a.show)


if __name__ == "__main__":
    main()
