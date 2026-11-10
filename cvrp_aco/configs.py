"""Algorithm configuration and the two presets used in the project.

PRESET_ACS_HYBRID : Ant Colony System + local search (the method described in
                    the "ACS" version of the Deliverable 1 notebook).
PRESET_AS_ELITIST : the Elitist Ant System of the team's first prototype
                    (the method described in the "Ant System" version of the notebook).
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ACOConfig:
    # --- variant ---------------------------------------------------------------
    variant: str = "acs"             # "acs" (Ant Colony System) or "as" (elitist Ant System)
    # --- construction ------------------------------------------------------------
    num_ants: int = 15
    alpha: float = 1.0               # influence of the pheromone
    beta: float = 2.0                # influence of the visibility eta = 1 / cost
    candidate_size: int | None = 20  # size of the candidate lists (None = all customers)
    # --- ACS specific ------------------------------------------------------------
    q0: float = 0.9                  # probability of the greedy choice (exploitation)
    xi: float = 0.1                  # local pheromone update rate
    rho: float = 0.1                 # evaporation / global update rate
    # --- AS specific ---------------------------------------------------------------
    deposit: float = 100.0           # pheromone constant (called Q in the first prototype)
    elitist_weight: float = 2.0      # extra reinforcement of the best-so-far solution
    tau_init: float = 1.0
    tau_min: float = 1e-8
    tau_max: float = 1e6
    # --- local search ---------------------------------------------------------------
    ls_scope: str = "all"             # "none" | "iteration_best" | "all"
    ls_neighbors: int = 10            # neighbors examined by relocate / swap
    # --- stopping criteria ------------------------------------------------------------
    max_iterations: int = 100
    patience: int | None = 30         # stop after this many iterations without improvement
    time_limit: float | None = None   # CPU seconds
    # --- fleet size -------------------------------------------------------------------
    penalty_factor: float = 10.0      # penalty = factor * max cost per route above k
    init_with_greedy: bool = True     # start from the Clarke and Wright solution

    def but(self, **kw):
        """Return a copy with some fields replaced."""
        return replace(self, **kw)


PRESET_ACS_HYBRID = ACOConfig()

PRESET_AS_ELITIST = ACOConfig(
    variant="as", num_ants=20, alpha=1.0, beta=2.0, rho=0.5, deposit=100.0,
    elitist_weight=2.0, tau_init=1.0, candidate_size=None, q0=0.0, xi=0.0,
    ls_scope="none", max_iterations=100, patience=None, init_with_greedy=False,
)
