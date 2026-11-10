"""Instance creation: data structure, random generators, VRPLIB loader.

Data structure (see the `Instance` class)
-----------------------------------------
* `cost`       : (n+1) x (n+1) integer NumPy matrix, constant-time access to c_ij.
                 It is the *completed* matrix (shortest-path costs), see section 4.9
                 of the notebook.
* `cost_list`  : the same matrix as a list of lists (faster than NumPy for scalar access).
* `adjacency`  : adjacency lists of the ORIGINAL graph (tuples of neighbors).
* `candidates` : for each vertex, the tuple of its nearest customers (candidate lists).
* `demands`    : NumPy vector with demands[0] = 0.
"""
import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path


@dataclass(eq=False)
class Instance:
    name: str
    coords: np.ndarray | None
    demands: np.ndarray
    capacity: int
    num_vehicles: int
    cost: np.ndarray
    adjacency: list
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n = len(self.demands) - 1                       # number of customers
        self.cost_list = self.cost.tolist()
        self.demand_list = [int(d) for d in self.demands]
        self._cand_cache = {}

    # ------------------------------------------------------------------ properties
    @property
    def total_demand(self):
        return int(self.demands.sum())

    @property
    def tightness(self):
        """total demand / (k * capacity): close to 1 means a binding capacity."""
        return self.total_demand / (self.num_vehicles * self.capacity)

    @property
    def min_routes(self):
        return math.ceil(self.total_demand / self.capacity)

    @property
    def density(self):
        """Share of the possible edges that exist in the original graph."""
        edges = sum(len(a) for a in self.adjacency) / 2
        return edges / (self.n * (self.n + 1) / 2)

    @property
    def mean_degree(self):
        return float(np.mean([len(a) for a in self.adjacency]))

    def candidates(self, size):
        """Candidate lists: for each vertex, the `size` nearest customers (tuple of ints)."""
        if size is None or size >= self.n:
            size = self.n
        if size not in self._cand_cache:
            order = np.argsort(self.cost, axis=1, kind="stable")
            lists = []
            for i in range(self.n + 1):
                near = [int(j) for j in order[i] if j != i and j != 0][:size]
                lists.append(np.array(near, dtype=int))
            self._cand_cache[size] = lists
        return self._cand_cache[size]

    def summary(self):
        return (f"{self.name}: n={self.n}, k={self.num_vehicles}, Q={self.capacity}, "
                f"tightness={self.tightness:.2f}, density={self.density:.2f}")


# ---------------------------------------------------------------------------------
def euclidean_cost(coords):
    """Rounded Euclidean distances (vectorized)."""
    p = np.asarray(coords, dtype=float)
    diff = p[:, None, :] - p[None, :, :]
    return np.rint(np.sqrt((diff ** 2).sum(axis=2))).astype(int)


def _unique_points(rng, count, layout, grid):
    """`count` distinct integer points (no two customers at the same place)."""
    if layout == "random":
        pts = rng.choice(grid * grid, size=count, replace=False)
        return np.column_stack((pts // grid, pts % grid))
    if layout == "clustered":
        centers = rng.integers(0, grid, size=(max(2, count // 25), 2))
        pts = set()
        while len(pts) < count:
            c = centers[rng.integers(len(centers))]
            p = np.clip(np.rint(c + rng.normal(0, grid / 15, 2)), 0, grid - 1).astype(int)
            pts.add((int(p[0]), int(p[1])))
        arr = np.array(sorted(pts))
        return arr[rng.permutation(count)]
    raise ValueError(f"unknown layout: {layout}")


def _sparse_graph(cost, density):
    """Keep the m nearest neighbors of each vertex + a spanning tree (connected graph)."""
    size = cost.shape[0]
    m = max(3, int(round(density * (size - 1))))
    keep = np.zeros_like(cost, dtype=bool)
    order = np.argsort(cost, axis=1, kind="stable")
    for i in range(size):
        for j in [v for v in order[i] if v != i][:m]:
            keep[i, j] = keep[j, i] = True
    mst = minimum_spanning_tree(csr_matrix(cost)).toarray() > 0
    keep |= mst | mst.T
    np.fill_diagonal(keep, False)
    return keep


def generate_instance(name, n, layout="random", tightness=0.9, capacity=30, density=1.0,
                      depot="random", demand_range=(1, 9), grid=100, seed=0):
    """Random CVRP instance.

    n          number of customers
    layout     "random" or "clustered"
    tightness  total demand / (k * capacity); k = ceil(total demand / (tightness * capacity))
    density    share of kept edges (1.0 = complete graph; smaller = sparse road-like graph,
               which is then completed with shortest paths)
    depot      "random", "center" or "corner"
    """
    rng = np.random.default_rng(seed)
    coords = _unique_points(rng, n + 1, layout, grid)
    if depot != "random":
        target = np.array([grid // 2, grid // 2] if depot == "center" else [0, 0])
        hit = np.flatnonzero((coords == target).all(axis=1))
        if len(hit):                                    # keep every point distinct
            coords[hit[0]] = coords[0]
        coords[0] = target
    demands = np.concatenate(([0], rng.integers(demand_range[0], demand_range[1] + 1, size=n)))
    k = math.ceil(demands.sum() / (tightness * capacity))

    full = euclidean_cost(coords)
    if density >= 1.0:
        cost = full
        adjacency = [tuple(j for j in range(n + 1) if j != i) for i in range(n + 1)]
    else:
        keep = _sparse_graph(full, density)
        graph = csr_matrix(np.where(keep, full, 0))
        cost = np.rint(shortest_path(graph, method="D", directed=False)).astype(int)
        adjacency = [tuple(int(j) for j in np.flatnonzero(keep[i])) for i in range(n + 1)]
    meta = dict(layout=layout, depot=depot, seed=seed, requested_density=density)
    return Instance(name, coords, demands, capacity, k, cost, adjacency, meta)


def load_vrplib(path, num_vehicles=None):
    """Load a VRPLIB instance (needs the `vrplib` package).

    Distances are rounded to the nearest integer (convention of the X instances).
    The number of vehicles is read from the name (...-k25) when possible.
    """
    import vrplib
    raw = vrplib.read_instance(str(path))
    demands = np.asarray(raw["demand"], dtype=int)
    capacity = int(raw["capacity"])
    cost = np.rint(np.asarray(raw["edge_weight"], dtype=float)).astype(int)
    name = str(raw.get("name", path))
    if num_vehicles is None:
        found = re.search(r"-k(\d+)", name)
        num_vehicles = int(found.group(1)) if found else math.ceil(demands.sum() / capacity) + 2
    size = len(demands)
    adjacency = [tuple(j for j in range(size) if j != i) for i in range(size)]
    coords = np.asarray(raw["node_coord"]) if "node_coord" in raw else None
    return Instance(name, coords, demands, capacity, int(num_vehicles), cost, adjacency,
                    dict(layout="vrplib", source=str(path)))


def best_known_cost(solution_path):
    """Cost of the best known solution (.sol file of VRPLIB)."""
    import vrplib
    return float(vrplib.read_solution(str(solution_path))["cost"])
