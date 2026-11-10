import numpy as np
import matplotlib.pyplot as plt
import random
import time

# ---Line 126 and 237 for change the scneario (dataset)---

# Generation randomly dataset of the coordinates of depot + customer and demands
def generate_cvrp_dataset(
    num_customers,
    num_vehicles,
    capacity,
    seed=None # seed for the same random result every time (Reproducibility)
):
    if seed is not None:
        random.seed(seed)

    # Depot + Customer coordinates
    coordinates = [
        (random.randint(0, 35), random.randint(0, 35)) # Range of 0 - 35 for the travel cost do not exceed 50
        for _ in range(num_customers + 1)
    ]

    # Depot demand = 0
    demands = [0]

    # Generate customer demands range of 1 ~ 5
    for _ in range(num_customers):
        demands.append(random.randint(1, 5)) 

    # Check whether total demand can be served
    total_demand = sum(demands)

    # If the vehicle's capacity exceeds total demands
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
    (5, 2, 15), # 5 customers, 2 vehicules, 15 capacity per vehicule
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

# An empty list to store the generated datasets in order
datasets = []

# Iterate through the predefined scenarios one by one, using "enumerate" to retrieve the index "i" that indicates the current iteration number
for i, (customers, vehicles, capacity) in enumerate(scenarios):

    dataset = generate_cvrp_dataset(
        num_customers=customers,
        num_vehicles=vehicles,
        capacity=capacity,
        seed=42 + i # 42 + 0 = seed 42, 42 + 1 = seed 43, 42 + 2 = seed 44...
    )

    datasets.append(dataset)


# Travel costs with calculation of Euclidean distance
def generate_cost_matrix(coordinates):

    num_vertices = len(coordinates)

    # Cost matrix matrix filled with zeros
    cost_matrix = np.zeros(
        (num_vertices, num_vertices),
        dtype=int
    )

    for i in range(num_vertices):
        for j in range(num_vertices):

            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]

            # Euclidean distance between two customers (two vertices)
            distance = np.sqrt(
                (x1 - x2)**2 + (y1 - y2)**2
            )

            cost_matrix[i][j] = round(distance) # round to make an integer

    return cost_matrix


# Generate Cost Matrix for each dataset
for dataset in datasets:

    dataset["cost_matrix"] = generate_cost_matrix(
        dataset["coordinates"]
    )


# Display a dataset we want (between 0 - 19 -> 20 scenairos) (line 51)
example = datasets[2]

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


# Visualize the graph we want by changing the dataset index
visualize_cvrp_graph(datasets[2])