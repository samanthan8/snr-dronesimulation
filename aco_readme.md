# Drone Simulation using Ant Colony Optimization (ACO) in PyBullet

This project simulates a swarm of drones using a modified Ant Colony Optimization (ACO) algorithm in a PyBullet physics environment. The drones try to find a goal location by following pheromone trails, similar to how ants find food sources.

## Ant Colony Optimization (ACO) Algorithm Overview

ACO is a metaheuristic optimization algorithm inspired by the foraging behavior of ants. In nature, ants deposit pheromones on the ground as they travel, and the pheromone concentration guides other ants towards promising paths. The basic principles of ACO are:

1.  **Pheromone Trails:** Ants (or in this case, drones) deposit pheromone on their paths. The amount of pheromone deposited can be related to the quality of the solution found (e.g., shorter paths get more pheromone).

2.  **Probabilistic Path Selection:** Ants choose their next move based on a probability function that considers both the pheromone concentration on neighboring paths and a heuristic value (e.g., distance to the goal).

3.  **Pheromone Evaporation:** Pheromone trails evaporate over time, preventing the algorithm from getting stuck in local optima and encouraging exploration.

**In the context of this drone simulation:**

*   **Ants:** Represent the drones.
*   **Pheromone:** A scalar value associated with discrete grid cells in the environment. Higher pheromone values indicate more desirable paths.
*   **Goal:** A fixed location in the environment that the drones are trying to reach.
*   **Heuristic:** The inverse of the distance to the goal.

## Code Structure

The code consists of the following main parts:

### Parameters

Defined at the top of the script, these parameters control the simulation and the ACO algorithm:

*   `NUM_DRONES`: The number of drones.
*   `NUM_ITERATIONS`: The number of simulation steps.
*   `EVAPORATION_RATE`: The rate at which pheromone evaporates in each iteration.
*   `ALPHA`: The importance of pheromone in the probability calculation.
*   `BETA`: The importance of the heuristic (distance to the goal) in the probability calculation.
*   `Q`: The amount of pheromone deposited by a drone that has found the goal.
*   `INITIAL_PHEROMONE`: The initial pheromone value for each grid cell.
*   `DRONE_SPEED`: The maximum speed of the drones.
*   `SEARCH_RADIUS`: The radius used for collision checks.
*   `GRID_SIZE`: The size of each cell in the pheromone grid.
*   `HOVER_HEIGHT`: The height at which the drones try to fly.
*   `ARENA_SIZE`: The size of the square simulation area.

### Environment Setup

*   Initializes the PyBullet physics engine (`p.connect(p.GUI)`).
*   Sets the gravity (`p.setGravity()`).
*   Loads a ground plane (`p.loadURDF("plane.urdf")`).
*   Defines the `start_pos` (where drones start) and `goal_pos`.
*   Visualizes the start and goal positions with red and green lines, respectively.
*   Creates obstacles using the `create_obstacle()` function.

### Pheromone Grid

*   `grid_x_min`, `grid_x_max`, `grid_y_min`, `grid_y_max`, `grid_z_min`, `grid_z_max`: Define the boundaries of the 3D grid.
*   `pheromone_grid`: A dictionary that stores the pheromone value for each grid cell. The keys are tuples `(x, y, z)` representing the cell coordinates.
*   `pheromone_line_ids`: A dictionary to store the IDs of the lines used to visualize pheromone levels, which allows efficient updating of the lines.
*   `id_offset`: An offset added to cell coordinates when generating unique IDs for pheromone lines to avoid potential negative ID issues.
*   The code initializes the `pheromone_grid` with `INITIAL_PHEROMONE` and visualizes the grid at `HOVER_HEIGHT` using lines. The color intensity of the lines corresponds to the pheromone level.
*   `get_cell_from_pos(pos)`: A function that converts a continuous 3D position to the corresponding grid cell coordinates.

### Drone Class

This class represents a single drone and implements the ACO logic:

*   `__init__(self, id, start_pos)`: Initializes the drone's ID, start position, path, and loads the quadrotor URDF model.
*   `calculate_probabilities(self)`: Calculates the probabilities of moving to neighboring cells based on pheromone levels and the heuristic (inverse distance to the goal). It also performs collision checks.
*   `choose_next_cell(self)`: Selects the next cell to move to based on the calculated probabilities using a roulette wheel selection method.
*   `move(self)`: Implements the drone's movement. If the drone has found the goal, it retraces its steps back to the start; otherwise, it chooses the next cell and moves towards it using `calculate_movement()`.
*   `calculate_movement(self, target_pos)`: Calculates a smooth movement vector towards the target position, considering the drone's speed.
*   `is_collision(self, next_pos)`: Checks if moving to `next_pos` would cause a collision with any obstacle.
*   `deposit_pheromone(self)`: Deposits pheromone on the cells in the drone's path if it has found the goal. The amount of pheromone deposited is inversely proportional to the path length.

### Initialization

*   Creates a list of `Drone` objects, all starting at `start_pos`.

### Main Loop

This loop runs the simulation for `NUM_ITERATIONS` steps:

1.  **Move Drones:** Each drone decides on its next move and updates its position.
2.  **Deposit Pheromone:** Drones that have reached the goal deposit pheromone along their paths.
3.  **Evaporate Pheromone:** The pheromone level in each cell is reduced by the `EVAPORATION_RATE`.
4.  **Update Visualization:** The pheromone grid visualization is updated to reflect the changes in pheromone levels.
5.  **Physics Simulation:** `p.stepSimulation()` advances the PyBullet physics simulation.
6.  **Visualization:** The simulation is visualized in the PyBullet GUI. `time.sleep()` controls the speed.

## Running the Simulation

### Prerequisites

*   Python 3.x
*   PyBullet
*   Gymnasium
*   NumPy

### Installation

```bash
pip install gymnasium pybullet numpy