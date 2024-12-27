import gymnasium as gym
import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import csv
import os

# --- Parameters ---
NUM_ITERATIONS = 500
C1 = 1.5  # Cognitive parameter
C2 = 1.5  # Social parameter
W = 0.7  # Inertia weight
V_MAX = 0.1  # Maximum velocity
DRONE_RADIUS = 0.1
HOVER_HEIGHT = 1.0
ARENA_SIZE = 5
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
    noise = np.random.normal(0, SIGNAL_NOISE_STD)
    noisy_signal_strength = signal_strength + noise
    return max(0, noisy_signal_strength)

# --- Drone Class ---
class Drone:
    def __init__(self, id, start_pos):
        self.id = id
        self.pos = np.array(start_pos, dtype=np.float64)
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.best_pos = self.pos.copy()
        self.best_signal = float('-inf')
        self.body = p.loadURDF(quadrotor_urdf_path, self.pos, globalScaling=DRONE_RADIUS*10)
        self.target_found = False
        self.target_found_in_iteration = -1

    def calculate_fitness(self, target_pos):
        return self.sense_signal(target_pos)

    def update_personal_best(self, target_pos):
        signal_strength = self.calculate_fitness(target_pos)
        if signal_strength > self.best_signal:
            self.best_signal = signal_strength
            self.best_pos = self.pos.copy()

    def update_velocity(self, global_best_pos):
        r1 = np.random.rand(3)
        r2 = np.random.rand(3)
        cognitive_component = C1 * r1 * (self.best_pos - self.pos)
        social_component = C2 * r2 * (global_best_pos - self.pos)
        self.velocity = W * self.velocity + cognitive_component + social_component
        self.velocity = np.clip(self.velocity, -V_MAX, V_MAX)

    def move(self, iteration, target_pos):
        self.pos += self.velocity

        # Check if target is found
        if np.linalg.norm(self.pos - np.array(target_pos)) < TARGET_REACH_THRESHOLD:
            self.target_found = True
            self.target_found_in_iteration = iteration
            print(f"Target found by drone {self.id} in iteration {self.target_found_in_iteration}!")
            self.pos = np.array(target_pos)

        # Check for collisions
        if self.is_collision():
            self.pos -= self.velocity
            self.velocity = -0.5 * self.velocity

        # Keep the drone within the arena bounds
        self.pos[0] = np.clip(self.pos[0], -ARENA_SIZE, ARENA_SIZE)
        self.pos[1] = np.clip(self.pos[1], -ARENA_SIZE, ARENA_SIZE)
        self.pos[2] = np.clip(self.pos[2], 0, ARENA_SIZE)

        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))

    def is_collision(self):
        for obstacle_id in obstacles:
            closest_points = p.getClosestPoints(self.body, obstacle_id, distance=DRONE_RADIUS*2)
            if closest_points:
                return True
        return False

    def sense_signal(self, target_pos):
        return get_signal_strength(self.pos, target_pos)

# --- Create Results Directory ---
RESULTS_DIR = "pso_simulation_results_csv"
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
    results_file = os.path.join(RESULTS_DIR, f"pso_num_drones_{num_drones}.csv")
    print(f"Running PSO with {num_drones} drones...")

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

            global_best_pos = drones[0].pos.copy()
            global_best_signal = float('-inf')
            target_found = False
            iterations_to_find_target = 0

            for iteration in range(NUM_ITERATIONS):
                # Update personal and global bests
                for drone in drones:
                    drone.update_personal_best(target_pos)
                    if drone.best_signal > global_best_signal:
                        global_best_signal = drone.best_signal
                        global_best_pos = drone.best_pos.copy()

                # Update velocity and move drones
                for drone in drones:
                    drone.update_velocity(global_best_pos)
                    drone.move(iteration, target_pos)

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