# Drone Simulation using Glowworm Swarm Optimization (GSO) in PyBullet

This project simulates a swarm of drones using the Glowworm Swarm Optimization (GSO) algorithm in a PyBullet physics environment. The drones are programmed to find a target location (representing a light source) by mimicking the behavior of glowworms, where they are attracted to other glowworms with higher "luciferin" levels.

## Glowworm Swarm Optimization (GSO) Algorithm Overview

GSO is a swarm intelligence algorithm inspired by the flashing behavior of glowworms. Each glowworm carries a luminescent quantity called "luciferin." The algorithm is based on the following principles:

1.  **Luciferin Update:** Each glowworm's luciferin level changes based on its current position and the objective function of the problem. Glowworms in better positions (closer to the target in this case) have their luciferin enhanced, while it decays over time for all glowworms.

2.  **Movement Decision:** Glowworms are attracted to neighbors that have a higher luciferin value than their own. They move towards a chosen neighbor within a certain sensor range (neighborhood).

3.  **Neighborhood Range Update:** The sensor range (also known as the decision radius or local-decision domain) of each glowworm is adaptive. If a glowworm has few neighbors, it increases its range to find more; if it has many neighbors, it reduces its range to focus on the most promising ones.

**In the context of this drone simulation:**

*   **Glowworms:** Represent the drones.
*   **Luciferin:** A value associated with each drone, indicating its proximity to the target (higher luciferin means closer to the target).
*   **Target:** A fixed location in the environment that represents the "light source" the drones are trying to find.
*   **Objective Function:** The inverse of the distance to the target. The goal is to maximize this function.

## Code Structure

The code consists of the following main parts:

### Parameters

Defined at the top of the script, these parameters control the simulation and the GSO algorithm:

*   `NUM_DRONES`: The number of drones in the simulation.
*   `NUM_ITERATIONS`: The number of simulation steps.
*   `INITIAL_LUCIFERIN`: The initial luciferin value for each drone.
*   `LUCIFERIN_DECAY_RATE`: The rate at which luciferin decays in each iteration.
*   `LUCIFERIN_ENHANCE_RATE`: The rate at which the objective function influences the luciferin increase.
*   `GAMMA`: A parameter for updating the sensor range.
*   `BETA`: A constant used in the sensor range update.
*   `S`: The step size for drone movement.
*   `RS`: The initial sensor range.
*   `DRONE_SPEED`: Maximum speed of the drones (currently not directly used in movement, but could be in more advanced implementations).
*   `HOVER_HEIGHT`: The height at which drones try to hover.
*   `ARENA_SIZE`: The size of the square simulation area.
*   `MAX_NEIGHBORS`: The maximum number of neighbors a drone considers.
*   `NT`: The desired number of neighbors.

### Environment Setup

*   Initializes the PyBullet physics engine (`p.connect(p.GUI)`).
*   Sets the gravity (`p.setGravity()`).
*   Loads a ground plane (`p.loadURDF("plane.urdf")`).
*   Defines the `target_pos` (the goal location).
*   Visualizes the target as a yellow sphere.
*   Creates some obstacles using `create_obstacle()`.

### Drone Class

This class represents a single drone and implements the GSO logic:

*   `__init__(self, id, start_pos)`: Initializes the drone's ID, position, luciferin level, sensor range, and loads the quadrotor URDF model.
*   `calculate_objective_function(self)`: Calculates the drone's objective function value (inverse distance to the target).
*   `update_luciferin(self)`: Updates the drone's luciferin level based on the objective function and the decay and enhancement rates.
*   `find_neighbors(self, drones)`: Finds neighbors within the sensor range that have higher luciferin values and returns a list of those neighbors.
*   `move_towards_neighbors(self, neighbors)`: Calculates the movement direction based on the weighted average of directions towards neighbors with higher luciferin. It then moves the drone in that direction and handles basic collision avoidance and boundary checks.
*   `update_sensor_range(self, neighbors)`: Updates the sensor range based on the number of neighbors found.
*   `is_collision(self, next_pos)`: Checks if the drone would collide with any obstacle at a given position.

### Initialization

*   Creates a list of `Drone` objects with random starting positions within the arena.

### Main Loop

This loop runs the simulation for `NUM_ITERATIONS` steps:

1.  **Luciferin Update:** Each drone's luciferin level is updated based on its current position.
2.  **Neighborhood Update:** Each drone finds its neighbors and updates its sensor range.
3.  **Movement:** Each drone moves towards its neighbors with a higher luciferin level.
4.  **Physics Simulation:** `p.stepSimulation()` advances the PyBullet physics simulation.
5.  **Visualization:** The simulation is visualized in the PyBullet GUI. `time.sleep()` controls the speed.

## Running the Simulation

### Prerequisites

*   Python 3.x
*   PyBullet
*   Gymnasium
*   NumPy

### Installation

```bash
pip install gymnasium pybullet numpy