# Drone Simulation using Particle Swarm Optimization (PSO) in PyBullet

This project simulates a swarm of drones using the Particle Swarm Optimization (PSO) algorithm in a PyBullet physics environment. The drones work together to find a target location by adjusting their velocities based on their own experience and the experience of the swarm.

## Particle Swarm Optimization (PSO) Algorithm Overview

PSO is a population-based stochastic optimization algorithm inspired by the social behavior of bird flocking or fish schooling. The algorithm maintains a swarm of particles (in this case, drones), where each particle represents a potential solution to the optimization problem.

**Key Principles of PSO:**

1.  **Particles and Positions:** Each particle has a position in the search space that represents a potential solution.
2.  **Velocities:** Each particle has a velocity that determines its movement through the search space.
3.  **Personal Best (pbest):** Each particle remembers its best position so far (the position that yielded the highest fitness value). This is called the particle's personal best (pbest).
4.  **Global Best (gbest):** The best position found by any particle in the entire swarm is tracked as the global best (gbest).
5.  **Velocity Update:** In each iteration, particles update their velocities based on:
    *   **Inertia:** The tendency to keep moving in the current direction.
    *   **Cognitive Component:** The attraction towards the particle's own personal best position.
    *   **Social Component:** The attraction towards the global best position.
6.  **Position Update:** Particles update their positions by adding the updated velocity to their current position.

**In the context of this drone simulation:**

*   **Particles:** Represent the drones.
*   **Position:** The 3D coordinates of a drone.
*   **Velocity:** The 3D vector representing the drone's speed and direction.
*   **Target:** A fixed location in the environment that the drones are trying to find.
*   **Fitness Function:** The inverse of the distance to the target. The goal is to maximize this function (closer to the target is better).

## Code Structure

The code consists of the following main parts:

### Parameters

Defined at the top of the script, these parameters control the simulation and the PSO algorithm:

*   `NUM_DRONES`: The number of drones in the swarm.
*   `NUM_ITERATIONS`: The number of simulation steps.
*   `C1`: The cognitive parameter (influence of personal best).
*   `C2`: The social parameter (influence of global best).
*   `W`: The inertia weight.
*   `V_MAX`: The maximum velocity of the drones.
*   `DRONE_RADIUS`: Radius of each drone used for collision detection.
*   `HOVER_HEIGHT`: The desired hovering height for the drones.
*   `ARENA_SIZE`: The size of the square simulation area.

### Environment Setup

*   Initializes the PyBullet physics engine (`p.connect(p.GUI)`).
*   Sets the gravity (`p.setGravity()`).
*   Loads a ground plane (`p.loadURDF("plane.urdf")`).
*   Defines the `target_pos` (the goal location).
*   Visualizes the target as a yellow sphere.
*   Creates some obstacles using `create_obstacle()`.

### Drone Class

This class represents a single drone and implements the PSO logic:

*   `__init__(self, id, start_pos)`: Initializes the drone's ID, position, velocity, personal best position, best fitness, and loads the quadrotor URDF model.
*   `calculate_fitness(self)`: Calculates the drone's fitness based on its distance to the target.
*   `update_personal_best(self)`: Updates the drone's personal best position and fitness if its current position is better.
*   `update_velocity(self, global_best_pos)`: Updates the drone's velocity based on inertia, cognitive, and social components, and limits the velocity to `V_MAX`.
*   `move(self)`: Updates the drone's position based on its velocity, handles basic collision avoidance with obstacles, and keeps the drone within the arena bounds.
*   `is_collision(self)`: Checks for collisions with obstacles.

### Initialization

*   Creates a list of `Drone` objects with random starting positions within the arena and zero initial velocities.
*   Initializes the `global_best_pos` and `global_best_fitness`.

### Main Loop

This loop runs the simulation for `NUM_ITERATIONS` steps:

1.  **Update Personal and Global Bests:** Each drone updates its personal best, and the global best position and fitness are updated.
2.  **Update Velocities and Move Drones:** Each drone updates its velocity based on the PSO algorithm and then moves to its new position.
3.  **Physics Simulation:** `p.stepSimulation()` advances the PyBullet physics simulation.
4.  **Visualization:** The simulation is visualized in the PyBullet GUI. `time.sleep()` controls the speed.

## Running the Simulation

### Prerequisites

*   Python 3.x
*   PyBullet
*   Gymnasium
*   NumPy

### Installation

```bash
pip install gymnasium pybullet numpy