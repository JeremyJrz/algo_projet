import numpy as np
import matplotlib.pyplot as plt
import random
import time

# ---Line 236 and 705 for change the scnearios (dataset)---

# Generation randomly dataset of the coordinates of customers and demands
def generate_cvrp_dataset(
    num_customers,
    num_vehicles,
    capacity,
    seed=None
):
    if seed is not None:
        random.seed(seed)

    # Depot + Customer coordinates
    coordinates = [
        (random.randint(0, 35), random.randint(0, 35))
        for _ in range(num_customers + 1)
    ]

    # Depot demand = 0
    demands = [0]

    # Generate customer demands
    for _ in range(num_customers):
        demands.append(random.randint(1, 5))

    # Check whether total demand can be served
    total_demand = sum(demands)

    if total_demand > num_vehicles * capacity:
        raise ValueError("Total demand exceeds total vehicle capacity.")

    # Generate the CVRP dataset
    dataset = {
        "num_customers": num_customers,
        "num_vehicles": num_vehicles,
        "capacity": capacity,
        "coordinates": coordinates,
        "demands": demands
    }

    return dataset


# Number of customers, vehicles, capacity of vehicles
scenarios = [
    (5, 2, 15),
    (8, 2, 20),
    (10, 3, 20),
    (12, 3, 25),
    (15, 3, 30),
    (18, 4, 30),
    (20, 4, 35),
    (22, 4, 40),
    (25, 5, 35),
    (28, 5, 40),
    (30, 5, 45),
    (35, 6, 45),
    (40, 6, 50),
    (45, 7, 50),
    (50, 7, 55),
    (55, 8, 55),
    (60, 8, 60),
    (70, 9, 65),
    (80, 10, 70),
    (100, 12, 75)
]


datasets = []

for i, (customers, vehicles, capacity) in enumerate(scenarios):

    dataset = generate_cvrp_dataset(
        num_customers=customers,
        num_vehicles=vehicles,
        capacity=capacity,
        seed=42 + i
    )

    datasets.append(dataset)


print("Number of datasets:", len(datasets))


# Travel costs with calculation of Euclidean distance
def generate_cost_matrix(coordinates):

    num_vertices = len(coordinates)

    cost_matrix = np.zeros(
        (num_vertices, num_vertices),
        dtype=int
    )

    for i in range(num_vertices):
        for j in range(num_vertices):

            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]

            distance = np.sqrt(
                (x1 - x2)**2 + (y1 - y2)**2
            )

            cost_matrix[i][j] = round(distance)

    return cost_matrix


# Generate Cost Matrix for each dataset
for dataset in datasets:

    dataset["cost_matrix"] = generate_cost_matrix(
        dataset["coordinates"]
    )


# Display an example dataset
example = datasets[0]

print("Customers:", example["num_customers"])
print("Vehicles:", example["num_vehicles"])
print("Capacity:", example["capacity"])

print("Coordinates:")
print(example["coordinates"])

print("Demands:")
print(example["demands"])

print("Cost Matrix:")
print(example["cost_matrix"])


# Graph visualization with Travel Costs
def visualize_cvrp_graph(dataset):

    coordinates = dataset["coordinates"]
    demands = dataset["demands"]
    cost_matrix = dataset["cost_matrix"]

    plt.figure(figsize=(12, 9))

    # Draw edges and display travel costs
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):

            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]

            # Draw edge
            plt.plot(
                [x1, x2],
                [y1, y2],
                color="gray",
                alpha=0.4,
                linewidth=1
            )

            # Calculate the midpoint of the edge
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Get travel cost
            travel_cost = cost_matrix[i][j]

            # Display travel cost
            plt.text(
                mid_x,
                mid_y,
                str(travel_cost),
                fontsize=8,
                color="black",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.8,
                    pad=1
                )
            )

    # Draw vertices
    for i, (x, y) in enumerate(coordinates):

        if i == 0:
            plt.scatter(
                x, y,
                color="red",
                s=180,
                label="Depot",
                zorder=3
            )

            plt.text(
                x + 0.5,
                y + 0.5,
                "Depot (0)",
                fontsize=10,
                fontweight="bold"
            )

        else:
            plt.scatter(
                x, y,
                color="blue",
                s=90,
                zorder=3
            )

            plt.text(
                x + 0.5,
                y + 0.5,
                f"C{i} (q={demands[i]})",
                fontsize=9
            )

    plt.title("CVRP Undirected Weighted Complete Graph")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.xlim(-3, 39)
    plt.ylim(-3, 39)
    plt.legend()
    plt.grid(True)
    plt.show()


# Visualize the graph by changing the dataset index
visualize_cvrp_graph(datasets[19])


# Display information about all datasets
for i, dataset in enumerate(datasets):

    print(f"Dataset {i + 1}")
    print("Customers:", dataset["num_customers"])
    print("Vehicles:", dataset["num_vehicles"])
    print("Capacity:", dataset["capacity"])
    print("Total demand:", sum(dataset["demands"]))
    print("Graph size:", dataset["cost_matrix"].shape)
    print("Minimum travel cost:", dataset["cost_matrix"].min())
    print("Maximum travel cost:", dataset["cost_matrix"].max())
    print("-" * 30)




# ------- ACO --------

# Check whether a solution satisfies all CVRP constraints
def is_feasible_solution(solution, dataset):

    num_customers = dataset["num_customers"]
    num_vehicles = dataset["num_vehicles"]
    capacity = dataset["capacity"]
    demands = dataset["demands"]

    # Check the number of vehicles
    if len(solution) > num_vehicles:
        return False

    visited_customers = []

    for route in solution:

        # Each route must start and end at the depot
        if len(route) < 3:
            return False

        if route[0] != 0 or route[-1] != 0:
            return False

        customers = route[1:-1]

        # Depot cannot appear inside a route
        if 0 in customers:
            return False

        # Check customer indices
        if any(c < 1 or c > num_customers for c in customers):
            return False

        # Check vehicle capacity
        total_load = sum(demands[c] for c in customers)

        if total_load > capacity:
            return False

        visited_customers.extend(customers)

    # Every customer must be visited exactly once
    return sorted(visited_customers) == list(
        range(1, num_customers + 1)
    )


# Calculate the total cost of a solution
def calculate_total_cost(solution, cost_matrix):

    total_cost = 0

    for route in solution:

        for i in range(len(route) - 1):

            current_node = route[i]
            next_node = route[i + 1]

            total_cost += cost_matrix[current_node][next_node]

    return total_cost


# Construct a CVRP solution using one ant
def construct_ant_solution(
    dataset,
    pheromone,
    heuristic,
    alpha,
    beta,
    rng,
    max_restarts=50
):

    num_customers = dataset["num_customers"]
    num_vehicles = dataset["num_vehicles"]
    capacity = dataset["capacity"]
    demands = dataset["demands"]

    # Every customer must individually fit in a vehicle
    if any(demands[c] > capacity for c in range(1, num_customers + 1)):
        return None

    # Retry if a random construction produces an infeasible solution
    for _ in range(max_restarts):

        unvisited = set(range(1, num_customers + 1))
        solution = []

        # Construct routes for available vehicles
        for vehicle in range(num_vehicles):

            if not unvisited:
                break

            route = [0]
            current_node = 0
            current_load = 0

            while unvisited:

                # Select customers that fit in the remaining capacity
                candidates = [
                    customer
                    for customer in unvisited
                    if current_load + demands[customer] <= capacity
                ]

                # Return to depot if no customer can be added
                if not candidates:
                    break

                # Calculate transition probabilities
                candidates = np.array(candidates, dtype=int)

                pheromone_values = pheromone[current_node, candidates]
                heuristic_values = heuristic[current_node, candidates]

                probabilities = (
                    pheromone_values ** alpha
                    * heuristic_values ** beta
                )

                probability_sum = probabilities.sum()

                # Use uniform probabilities if necessary
                if (
                    not np.isfinite(probability_sum)
                    or probability_sum <= 0
                ):
                    probabilities = np.ones(len(candidates)) / len(candidates)

                else:
                    probabilities = probabilities / probability_sum

                # Randomly select the next customer
                next_customer = int(
                    rng.choice(candidates, p=probabilities)
                )

                # Update route and vehicle load
                route.append(next_customer)

                current_load += demands[next_customer]
                current_node = next_customer

                unvisited.remove(next_customer)

            # Return to depot
            if len(route) > 1:
                route.append(0)
                solution.append(route)

            # All customers have been visited
            if not unvisited:
                break

        # Validate the completed solution
        if not unvisited and is_feasible_solution(solution, dataset):
            return solution

    # No feasible solution found during the attempts
    return None

# Ant Colony Optimization for CVRP
def aco_cvrp(
    dataset,
    num_ants=20,
    num_iterations=100,
    alpha=1,
    beta=2,
    evaporation_rate=0.5,
    Q=100,
    seed=42
):

    cost_matrix = dataset["cost_matrix"]
    num_vertices = len(cost_matrix)

    # Check whether the dataset is feasible
    total_demand = sum(dataset["demands"])

    if total_demand > dataset["num_vehicles"] * dataset["capacity"]:
        raise ValueError("Total demand exceeds total vehicle capacity.")

    if any(
        demand > dataset["capacity"]
        for demand in dataset["demands"][1:]
    ):
        raise ValueError("A customer demand exceeds vehicle capacity.")

    rng = np.random.default_rng(seed)

    # Initialize pheromone matrix
    pheromone = np.ones(
        (num_vertices, num_vertices),
        dtype=float
    )

    # Calculate heuristic information
    # Shorter distances receive higher heuristic values
    heuristic = 1.0 / (cost_matrix.astype(float) + 1e-6)

    # No self-transition
    np.fill_diagonal(heuristic, 0)

    best_solution = None
    best_cost = float("inf")

    convergence_history = []

    # Main ACO loop
    for iteration in range(num_iterations):

        iteration_solutions = []

        # Each ant constructs one solution
        for ant in range(num_ants):

            solution = construct_ant_solution(
                dataset=dataset,
                pheromone=pheromone,
                heuristic=heuristic,
                alpha=alpha,
                beta=beta,
                rng=rng
            )

            if solution is not None:

                solution_cost = calculate_total_cost(
                    solution,
                    cost_matrix
                )

                iteration_solutions.append(
                    (solution, solution_cost)
                )

                # Update global best solution
                if solution_cost < best_cost:

                    best_cost = solution_cost

                    best_solution = [
                        route.copy()
                        for route in solution
                    ]

        # Evaporate pheromone
        pheromone *= (1 - evaporation_rate)

        # Update pheromone using feasible solutions
        for solution, solution_cost in iteration_solutions:

            deposit = Q / max(solution_cost, 1e-9)

            for route in solution:

                for i in range(len(route) - 1):

                    node_a = route[i]
                    node_b = route[i + 1]

                    pheromone[node_a][node_b] += deposit
                    pheromone[node_b][node_a] += deposit

        # Additional reinforcement of the global best solution
        if best_solution is not None:

            best_deposit = 2 * Q / max(best_cost, 1e-9)

            for route in best_solution:

                for i in range(len(route) - 1):

                    node_a = route[i]
                    node_b = route[i + 1]

                    pheromone[node_a][node_b] += best_deposit
                    pheromone[node_b][node_a] += best_deposit

        # Prevent pheromone values from becoming excessively large
        pheromone = np.clip(
            pheromone,
            1e-8,
            1e6
        )

        # Record the best cost at each iteration
        convergence_history.append(best_cost)

        # Display progress
        if (iteration + 1) % 10 == 0:

            print(
                f"Iteration {iteration + 1}/{num_iterations} "
                f"| Best Cost: {best_cost}"
            )

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found. "
            "Try increasing the number of ants or construction attempts."
        )

    return best_solution, best_cost, convergence_history


# Visualize optimized CVRP routes with travel costs
def visualize_cvrp_solution(dataset, solution):

    coordinates = dataset["coordinates"]
    demands = dataset["demands"]
    cost_matrix = dataset["cost_matrix"]

    plt.figure(figsize=(12, 9))

    # Draw each vehicle route
    for vehicle_id, route in enumerate(solution):

        route_coordinates = [
            coordinates[node]
            for node in route
        ]

        x_values = [point[0] for point in route_coordinates]
        y_values = [point[1] for point in route_coordinates]

        # Draw route
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            label=f"Vehicle {vehicle_id + 1}",
            zorder=2
        )

        # Display travel cost on each traveled edge
        for i in range(len(route) - 1):

            node_a = route[i]
            node_b = route[i + 1]

            x1, y1 = coordinates[node_a]
            x2, y2 = coordinates[node_b]

            # Calculate midpoint of the edge
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Get travel cost from Cost Matrix
            travel_cost = cost_matrix[node_a][node_b]

            plt.text(
                mid_x,
                mid_y,
                str(travel_cost),
                fontsize=10,
                color="black",
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.9,
                    boxstyle="round,pad=0.25"
                ),
                zorder=3
            )

    # Draw depot
    depot_x, depot_y = coordinates[0]

    plt.scatter(
        depot_x,
        depot_y,
        color="red",
        s=200,
        marker="s",
        label="Depot",
        zorder=5
    )

    plt.text(
        depot_x + 0.5,
        depot_y + 0.5,
        "Depot (0)",
        fontsize=10,
        fontweight="bold"
    )

    # Draw customers
    for i in range(1, len(coordinates)):

        x, y = coordinates[i]

        plt.scatter(
            x,
            y,
            color="black",
            s=60,
            zorder=4
        )

        plt.text(
            x + 0.5,
            y + 0.5,
            f"C{i} (q={demands[i]})",
            fontsize=8
        )

    plt.title("ACO Optimized CVRP Routes with Travel Costs")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    plt.xlim(-3, 39)
    plt.ylim(-3, 39)

    plt.grid(True)
    plt.legend()
    plt.show()



# Visualize ACO convergence
def visualize_convergence(history):

    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, len(history) + 1),
        history,
        marker=".",
        linewidth=1.5
    )

    plt.title("ACO Convergence Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Best Total Travel Cost")

    plt.grid(True)
    plt.show()

# Select a dataset
dataset_index = 19

selected_dataset = datasets[dataset_index]

print("Selected Dataset:", dataset_index + 1)
print("Customers:", selected_dataset["num_customers"])
print("Vehicles:", selected_dataset["num_vehicles"])
print("Capacity:", selected_dataset["capacity"])

# Measure execution time
start_time = time.perf_counter()

# Run ACO
best_solution, best_cost, history = aco_cvrp(
    dataset=selected_dataset,
    num_ants=20,
    num_iterations=100,
    alpha=1,
    beta=2,
    evaporation_rate=0.5,
    Q=100,
    seed=42
)

end_time = time.perf_counter()

execution_time = end_time - start_time

# Display results
print("\n========== ACO RESULTS ==========")

for i, route in enumerate(best_solution):

    route_load = sum(
        selected_dataset["demands"][customer]
        for customer in route[1:-1]
    )

    print(f"Vehicle {i + 1}: {route}")
    print(f"Vehicle Load: {route_load}")
    print(f"Vehicle Capacity: {selected_dataset['capacity']}")

print("\nTotal Travel Cost:", best_cost)
print("Number of Vehicles Used:", len(best_solution))
print(f"Execution Time: {execution_time:.4f} seconds")

print(
    "Feasible Solution:",
    is_feasible_solution(best_solution, selected_dataset)
)

# Display graphs
visualize_cvrp_solution(
    selected_dataset,
    best_solution
)

visualize_convergence(history)