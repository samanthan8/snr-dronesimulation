import gymnasium as gym
import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import math
import json
import os

# --- Parameters ---
NUM_ITERATIONS = 500
EVAPORATION_RATE = 0.5
ALPHA = 1  # Importance of pheromone
BETA = 2   # Importance of heuristic (distance)
Q = 100  # Pheromone deposit amount
INITIAL_PHEROMONE = 1.0
DRONE_RADIUS = 0.1
SEARCH_RADIUS = 1.0
GRID_SIZE = 0.5
HOVER_HEIGHT = 1.0
ARENA_SIZE = 5
TARGET_REACH_THRESHOLD = 0.5
SIGNAL_FOUND_THRESHOLD = 0.7
SIGNAL_NOISE_STD = 0.05
NUM_RUNS = 30
NUM_DRONES_OPTIONS = [1, 5, 10, 15, 20, 25, 30]

# Define the path to the local directory where the URDF files are stored
LOCAL_URDF_PATH = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
# Define the URDF file paths
plane_urdf_path = os.path.join(LOCAL_URDF_PATH, "plane.urdf")  # Make sure 'plane.urdf' is in the same folder
quadrotor_urdf_path = os.path.join(LOCAL_URDF_PATH, "quadrotor.urdf")  # Make sure 'quadrotor.urdf' is in the same folder

# --- Environment Setup ---
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF(plane_urdf_path)

# --- Function to Generate Random Positions ---
def generate_random_start_pos():
    return [random.uniform(-ARENA_SIZE, ARENA_SIZE),
            random.uniform(-ARENA_SIZE, ARENA_SIZE),
            HOVER_HEIGHT]

def generate_random_target_pos():
    return [random.uniform(-ARENA_SIZE, ARENA_SIZE),
            random.uniform(-ARENA_SIZE, ARENA_SIZE),
            HOVER_HEIGHT]

# --- Create obstacles ---
def create_obstacle(pos, halfExtents, color=[0, 0, 1, 1]):
    colShape = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
    visShape = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=color)
    obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colShape, baseVisualShapeIndex=visShape, basePosition=pos)
    return obstacle

obstacles = []
obstacles.append(create_obstacle([0, 0, 0.5], [0.5, 0.5, 0.5]))
obstacles.append(create_obstacle([-1, 2, 1], [0.2, 0.2, 1], [0.5, 0.5, 1, 1]))
obstacles.append(create_obstacle([1.5, -1, 0.75], [0.7, 0.3, 0.75], [1, 0.5, 0, 1]))
obstacles.append(create_obstacle([3, 3, 0.5], [0.2, 0.2, 0.5]))
obstacles.append(create_obstacle([-3, -3, 0.5], [0.2, 0.2, 0.5]))

# --- Signal Strength Function with Noise ---
def get_signal_strength(drone_pos, target_pos):
    distance = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
    signal_strength = 1.0 / (distance ** 2 + 1e-6)

    # Add Gaussian noise
    noise = np.random.normal(0, SIGNAL_NOISE_STD)
    noisy_signal_strength = signal_strength + noise

    return max(0, noisy_signal_strength)

# --- Pheromone Grid ---
def initialize_pheromone_grid():
    grid_x_min, grid_x_max = -ARENA_SIZE, ARENA_SIZE
    grid_y_min, grid_y_max = -ARENA_SIZE, ARENA_SIZE
    grid_z_min, grid_z_max = 0, 3

    pheromone_grid = {}
    for x in np.arange(grid_x_min, grid_x_max + GRID_SIZE, GRID_SIZE):
        for y in np.arange(grid_y_min, grid_y_max + GRID_SIZE, GRID_SIZE):
            for z in np.arange(grid_z_min, grid_z_max + GRID_SIZE, GRID_SIZE):
                rounded_x, rounded_y, rounded_z = round(x, 1), round(y, 1), round(z, 1)
                pheromone_grid[(rounded_x, rounded_y, rounded_z)] = INITIAL_PHEROMONE
    return pheromone_grid

def get_cell_from_pos(pos):
    x = round(round(pos[0] / GRID_SIZE) * GRID_SIZE, 1)
    y = round(round(pos[1] / GRID_SIZE) * GRID_SIZE, 1)
    z = round(round(pos[2] / GRID_SIZE) * GRID_SIZE, 1)
    return (x, y, z)

# --- Drone Class ---
class Drone:
    def __init__(self, id, start_pos):
        self.id = id
        self.start_pos = start_pos
        self.pos = start_pos
        self.path = [get_cell_from_pos(start_pos)]
        self.path_positions = [start_pos]
        self.has_found_goal = False
        self.target_found = False
        self.target_found_in_iteration = -1
        self.body = p.loadURDF(quadrotor_urdf_path, self.pos, globalScaling=DRONE_RADIUS * 10)

    def calculate_probabilities(self, pheromone_grid, target_pos):
        current_cell = get_cell_from_pos(self.pos)
        probabilities = {}
        total = 0

        for dx in [-GRID_SIZE, 0, GRID_SIZE]:
            for dy in [-GRID_SIZE, 0, GRID_SIZE]:
                for dz in [-GRID_SIZE, 0, GRID_SIZE]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    next_cell = (
                        round(current_cell[0] + dx, 1),
                        round(current_cell[1] + dy, 1),
                        round(current_cell[2] + dz, 1)
                    )

                    if next_cell not in pheromone_grid:
                        continue

                    # Collision check
                    next_pos = [next_cell[0], next_cell[1], next_cell[2]]
                    if self.is_collision(next_pos):
                        continue

                    pheromone = pheromone_grid[next_cell]
                    signal_strength = get_signal_strength(next_pos, target_pos)
                    heuristic = signal_strength

                    attractiveness = (pheromone ** ALPHA) * (heuristic ** BETA)
                    probabilities[next_cell] = attractiveness
                    total += attractiveness

        # Normalize probabilities
        for cell in probabilities:
            probabilities[cell] /= total

        return probabilities

    def choose_next_cell(self, pheromone_grid, target_pos):
        probabilities = self.calculate_probabilities(pheromone_grid, target_pos)
        if not probabilities:
            return get_cell_from_pos(self.pos)

        rand_num = random.random()
        cumulative_prob = 0
        for cell, prob in probabilities.items():
            cumulative_prob += prob
            if rand_num <= cumulative_prob:
                return cell

    def move(self, iteration, pheromone_grid, target_pos):
        if self.has_found_goal:
            if len(self.path_positions) > 1:
                next_pos = self.path_positions[-2]
                self.path_positions.pop()
                self.pos = self.calculate_movement(next_pos)
            else:
                self.pos = self.calculate_movement(self.start_pos)
                self.path = [get_cell_from_pos(self.start_pos)]
                self.path_positions = [self.start_pos]
                self.has_found_goal = False
        else:
            next_cell = self.choose_next_cell(pheromone_grid, target_pos)
            target_pos_for_move = [next_cell[0], next_cell[1], next_cell[2]]

            if get_signal_strength(self.pos, target_pos) > SIGNAL_FOUND_THRESHOLD:
                self.has_found_goal = True
                self.target_found_in_iteration = iteration
                print(f"Target found by drone {self.id} in iteration {self.target_found_in_iteration}!")
                self.path.append(get_cell_from_pos(target_pos))
                self.path_positions.append(target_pos)
                self.pos = target_pos
            else:
                self.path.append(next_cell)
                self.path_positions.append(target_pos_for_move)
                self.pos = self.calculate_movement(target_pos_for_move)

        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))

    def calculate_movement(self, target_pos):
        direction = np.array(target_pos) - np.array(self.pos)
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance

        # Limit the movement to a small step
        max_step = 0.05
        actual_movement = min(distance, max_step)
        new_pos = np.array(self.pos) + direction * actual_movement

        return list(new_pos)

    def is_collision(self, next_pos):
        p.resetBasePositionAndOrientation(self.body, next_pos, p.getQuaternionFromEuler([0, 0, 0]))
        for obstacle_id in obstacles:
            closest_points = p.getClosestPoints(self.body, obstacle_id, distance=DRONE_RADIUS + SEARCH_RADIUS)
            if closest_points:
                p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))
                return True
        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))
        return False

    def deposit_pheromone(self, pheromone_grid):
        if self.has_found_goal:
            path_length = len(self.path)
            pheromone_amount = Q / path_length  # You might want to base this on signal strength instead
            for cell in self.path:
                pheromone_grid[cell] += pheromone_amount

# --- Create Results Directory ---
RESULTS_DIR = "aco_simulation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Main Loop ---
start_positions = [[-4, -4, HOVER_HEIGHT],
                   [-4, 4, HOVER_HEIGHT],
                   [4, -4, HOVER_HEIGHT],
                   [4, 4, HOVER_HEIGHT],
                   [0, 0, HOVER_HEIGHT],
                   [-2, -2, HOVER_HEIGHT],
                   [-2, 2, HOVER_HEIGHT],
                   [2, -2, HOVER_HEIGHT],
                   [2, 2, HOVER_HEIGHT],
                   [0, 2, HOVER_HEIGHT],
                   [0, -2, HOVER_HEIGHT],
                   [-3, 0, HOVER_HEIGHT],
                   [3, 0, HOVER_HEIGHT],
                   [-1, 0, HOVER_HEIGHT],
                   [1, 0, HOVER_HEIGHT],
                   [0, 1, HOVER_HEIGHT],
                   [0, -1, HOVER_HEIGHT],
                   [-1, -1, HOVER_HEIGHT],
                   [-1, 1, HOVER_HEIGHT],
                   [1, -1, HOVER_HEIGHT],
                   [1, 1, HOVER_HEIGHT]]

for num_drones in NUM_DRONES_OPTIONS:
    results_file = os.path.join(RESULTS_DIR, f"aco_num_drones_{num_drones}.jsonl")
    print(f"Running ACO with {num_drones} drones...")

    for run in range(NUM_RUNS):
        print(f"  Run: {run + 1}")

        # Reinitialize PyBullet environment for each run
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        planeId = p.loadURDF(plane_urdf_path)

        # Generate random target position
        target_pos = generate_random_target_pos()

        # Initialize pheromone grid
        pheromone_grid = initialize_pheromone_grid()

        # Initialize drones
        drones = [Drone(i, start_positions[i%len(start_positions)]) for i in range(num_drones)]

        # Reset obstacles for each run
        # for obstacle in obstacles:
        #     p.removeBody(obstacle)
        obstacles = []
        obstacles.append(create_obstacle([0, 0, 0.5], [0.5, 0.5, 0.5]))
        obstacles.append(create_obstacle([-1, 2, 1], [0.2, 0.2, 1], [0.5, 0.5, 1, 1]))
        obstacles.append(create_obstacle([1.5, -1, 0.75], [0.7, 0.3, 0.75], [1, 0.5, 0, 1]))
        obstacles.append(create_obstacle([3, 3, 0.5], [0.2, 0.2, 0.5]))
        obstacles.append(create_obstacle([-3, -3, 0.5], [0.2, 0.2, 0.5]))

        target_found = False
        iterations_to_find_target = 0

        for iteration in range(NUM_ITERATIONS):
            # Move drones
            for drone in drones:
                drone.move(iteration, pheromone_grid, target_pos)

            # Deposit pheromone
            for drone in drones:
                drone.deposit_pheromone(pheromone_grid)

            # Evaporate pheromone
            for cell in pheromone_grid:
                pheromone_grid[cell] *= (1 - EVAPORATION_RATE)

            # Check if any drone has found the target
            for drone in drones:
                if drone.has_found_goal and drone.target_found_in_iteration != -1:
                    target_found = True
                    iterations_to_find_target = drone.target_found_in_iteration + 1
                    break

            # Simulation step
            p.stepSimulation()

            # Exit the loop if the target is found
            if target_found:
                break

        # Write results to file
        with open(results_file, "a") as f:
            result = {
                "run": run + 1,
                "num_drones": num_drones,
                "target_found": target_found,
                "iterations": iterations_to_find_target if target_found else NUM_ITERATIONS,
                "target_pos": list(target_pos)
            }
            for i, drone in enumerate(drones):
                result[f"drone_{i}_start_pos"] = list(start_positions[i%len(start_positions)])
                result[f"drone_{i}_found_in_iteration"] = drone.target_found_in_iteration if drone.target_found else -1
            f.write(json.dumps(result) + "\n")

# --- End Simulation ---
p.disconnect()