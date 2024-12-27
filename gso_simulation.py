import gymnasium as gym
import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import math
import json
import os
import csv

# --- Parameters ---
NUM_ITERATIONS = 500
INITIAL_LUCIFERIN = 5.0
LUCIFERIN_DECAY_RATE = 0.4
LUCIFERIN_ENHANCE_RATE = 0.6
GAMMA = 0.6
BETA = 0.08
S = 0.03
RS = 2.0
DRONE_RADIUS = 0.1
HOVER_HEIGHT = 1.0
ARENA_SIZE = 5
MAX_NEIGHBORS = 5
NT = 5
TARGET_REACH_THRESHOLD = 0.5
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

# --- Drone Class ---
class Drone:
    def __init__(self, id, start_pos):
        self.id = id
        self.pos = start_pos
        self.luciferin = INITIAL_LUCIFERIN
        self.sensor_range = RS
        self.body = p.loadURDF(quadrotor_urdf_path, self.pos, globalScaling=DRONE_RADIUS * 10)
        self.target_found = False
        self.target_found_in_iteration = -1

    def calculate_objective_function(self, target_pos):
        return self.sense_signal(target_pos)

    def update_luciferin(self, target_pos):
        self.luciferin = (1 - LUCIFERIN_DECAY_RATE) * self.luciferin + LUCIFERIN_ENHANCE_RATE * self.calculate_objective_function(target_pos)

    def find_neighbors(self, drones):
        neighbors = []
        for other in drones:
            if other.id != self.id:
                dist = np.linalg.norm(np.array(self.pos) - np.array(other.pos))
                if dist <= self.sensor_range and self.luciferin < other.luciferin:
                    neighbors.append((other, dist))
        neighbors.sort(key=lambda x: x[1])
        neighbors = neighbors[:MAX_NEIGHBORS]
        return [n[0] for n in neighbors]

    def move_towards_neighbors(self, neighbors, iteration, target_pos):
        if not neighbors:
            return

        total_weight = 0
        weighted_direction = np.array([0.0, 0.0, 0.0])
        for neighbor in neighbors:
            direction = np.array(neighbor.pos) - np.array(self.pos)
            distance = np.linalg.norm(direction)
            if distance > 1e-6:
                weight = (neighbor.luciferin - self.luciferin) / distance
                weighted_direction += weight * direction
                total_weight += weight

        if total_weight > 0:
            weighted_direction /= total_weight

        new_pos = np.array(self.pos) + S * weighted_direction

        # Check if target is reached
        if np.linalg.norm(new_pos - np.array(target_pos)) < TARGET_REACH_THRESHOLD:
            self.target_found = True
            self.target_found_in_iteration = iteration
            print(f"Target found by drone {self.id} in iteration {self.target_found_in_iteration}!")
            new_pos = np.array(target_pos)

        # Collision and boundary checks
        if self.is_collision(new_pos):
            self.pos = list(np.array(self.pos) - 0.5 * S * weighted_direction)
        else:
            self.pos = list(new_pos)

        self.pos[0] = np.clip(self.pos[0], -ARENA_SIZE, ARENA_SIZE)
        self.pos[1] = np.clip(self.pos[1], -ARENA_SIZE, ARENA_SIZE)
        self.pos[2] = np.clip(self.pos[2], 0, ARENA_SIZE)

        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))

    def update_sensor_range(self, neighbors):
        self.sensor_range = self.sensor_range + BETA * (NT - len(neighbors))
        self.sensor_range = max(0, min(self.sensor_range, RS))

    def is_collision(self, next_pos):
        p.resetBasePositionAndOrientation(self.body, next_pos, p.getQuaternionFromEuler([0, 0, 0]))
        for obstacle_id in obstacles:
            closest_points = p.getClosestPoints(self.body, obstacle_id, distance=DRONE_RADIUS*2)
            if closest_points:
                p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))
                return True
        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))
        return False

    def sense_signal(self, target_pos):
        return get_signal_strength(self.pos, target_pos)

# --- Create Results Directory ---
RESULTS_DIR = "gso_simulation_results_csv"
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
    results_file = os.path.join(RESULTS_DIR, f"gso_num_drones_{num_drones}.csv")
    print(f"Running GSO with {num_drones} drones...")

    with open(results_file, mode='w', newline='') as csvfile:
        fieldnames = ['run', 'num_drones', 'target_found', 'iterations', 'target_pos']
        for i in range(num_drones):
            fieldnames.extend([f'drone_{i}_start_pos', f'drone_{i}_found_in_iteration'])
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for run in range(NUM_RUNS):
            print(f"  Run: {run + 1}")

            # Reinitialize PyBullet environment for each run
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            planeId = p.loadURDF(plane_urdf_path)

            # Generate random target position
            target_pos = generate_random_target_pos()

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
                # Update luciferin for each drone
                for drone in drones:
                    drone.update_luciferin(target_pos)

                # Update sensor range and find neighbors for each drone
                for drone in drones:
                    neighbors = drone.find_neighbors(drones)
                    drone.update_sensor_range(neighbors)

                # Move drones towards neighbors
                for drone in drones:
                    neighbors = drone.find_neighbors(drones)
                    drone.move_towards_neighbors(neighbors, iteration, target_pos)

                # Check if any drone has found the target
                for drone in drones:
                    if drone.target_found and drone.target_found_in_iteration != -1:
                        target_found = True
                        iterations_to_find_target = drone.target_found_in_iteration + 1
                        break

                # Simulation step
                p.stepSimulation()

                # Exit the loop if the target is found
                if target_found:
                    break

            # Write results to file
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
            writer.writerow(result)

# --- End Simulation ---
p.disconnect()