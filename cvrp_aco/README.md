# CVRP – Ant colony optimization with local search (Group 3, A.C.O project)

Python package for the implementation and experimental phases of the project
(Capacitated Vehicle Routing Problem, level B of the statement).

## Files

| File | Role |
|---|---|
| `main.py` | command line entry point (`demo`, `experiments`, `sweep`, `vrplib`) |
| `instances.py` | data structure `Instance`, random instance generators (size, layout, capacity tightness, graph density, depot position), VRPLIB loader |
| `configs.py` | `ACOConfig` and the two presets: `PRESET_ACS_HYBRID` (ACS + local search) and `PRESET_AS_ELITIST` (elitist Ant System of the first prototype) |
| `aco.py` | main algorithm: ant construction, pheromone updates, stopping criteria (`run_aco`) |
| `local_search.py` | 2-opt, relocate, swap with incremental cost differences |
| `heuristics.py` | Clarke and Wright savings (baseline and initial solution) |
| `solution.py` | cost, load, feasibility check, edge arrays |
| `bounds.py` | lower bounds: degree bound and LP relaxation with capacity cuts |
| `experiments.py` | the 21 instance scenarios, runs over seeds, parameter sweeps, statistics (mean/std, median/MAD) |
| `plots.py` | routes, convergence curve, distribution of ant quality, box plots, time versus n |

## Installation

Python 3.10 or newer.

```
pip install -r requirements.txt
```

## Usage

```
python main.py demo                          # one instance, figures saved in outputs/
python main.py demo --preset as              # elitist Ant System (first prototype)
python main.py demo --preset as --ls all     # elitist Ant System + local search
python main.py experiments --quick           # small campaign (a few minutes)
python main.py experiments --seeds 10        # full campaign: 21 instances x 10 seeds
python main.py sweep --param rho --values 0.05 0.1 0.3 0.5
python main.py vrplib X-n101-k25.vrp --best-known X-n101-k25.sol
```

Results are written to `outputs/` (CSV files and figures). Add `--show` to open the figures.

## Design choices (to be justified in the notebook)

* The algorithms work on the **completed** cost matrix (shortest paths), also for sparse graphs
  (section 4.9 of the notebook).
* Both ACO variants build routes one after the other; a route is closed when no customer fits.
  If more than k routes are needed, the solution is penalized.
* The lower bound `lb_lp` is the LP relaxation with rounded capacity cuts separated heuristically:
  it is valid but not tight for large instances (use it as an indication, and compare also with
  Clarke and Wright, with Clarke and Wright + local search, and with the best known solutions).
* VRPLIB distances are rounded to the nearest integer (convention of the X instances).

## Known limitations

* Pure Python/NumPy: instances of several thousand customers need candidate lists
  (already implemented), a local search applied only to the best ant (`--ls iteration_best`),
  and possibly Numba for the local search.
* The VRPLIB loader was tested on a local file; check the field names with your version of `vrplib`.
* The ant colony alone adds a modest gain on top of Clarke and Wright + local search on some
  instances: this is measured by the columns `gain_vs_cw` and `gain_vs_cw_ls`.
