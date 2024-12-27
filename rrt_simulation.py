import gymnasium as gym
import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import csv
import os

# --- Parameters ---
NUM_ITERATIONS = 5000
DELTA = 0.5
COLLISION_CHECK_STEP = 0.05
DRONE_RADIUS = 0.1
HOVER_HEIGHT = 1.0
ARENA_SIZE = 5
SIGNAL_FOUND_THRESHOLD = 0.7
SEARCH_RADIUS = 0.5
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

# --- Function to Generate Random Start and Target Positions ---
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

# --- RRT Node Class ---
class Node:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.parent = None
        self.signal_strength = 0.0

# --- Drone Class ---
class Drone:
    def __init__(self, id, start_pos):
        self.id = id
        self.pos = np.array(start_pos, dtype=np.float64)
        self.body = p.loadURDF(quadrotor_urdf_path, start_pos, globalScaling=DRONE_RADIUS*10)
        self.path = []
        self.target_found = False
        self.target_found_in_iteration = -1
        self.tree = [Node(start_pos)]

    def move_along_path(self):
        if self.path:
            target_pos = self.path[0]
            direction = target_pos - self.pos
            distance = np.linalg.norm(direction)

            if distance > 1e-6:
                direction = direction / distance

            movement_step = 0.05

            if distance <= movement_step:
                self.pos = target_pos
                self.path.pop(0)
            else:
                self.pos = self.pos + direction * movement_step

            p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))

    def sense_signal(self, target_pos):
        return get_signal_strength(self.pos, target_pos)

    def is_collision(self, next_pos):
        p.resetBasePositionAndOrientation(self.body, next_pos, p.getQuaternionFromEuler([0, 0, 0]))
        for obstacle_id in obstacles:
            closest_points = p.getClosestPoints(self.body, obstacle_id, distance=DRONE_RADIUS + SEARCH_RADIUS)
            if closest_points:
                p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))
                return True
        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))
        return False

# --- RRT Functions ---
def get_random_config():
    return np.array([random.uniform(-ARENA_SIZE, ARENA_SIZE),
                     random.uniform(-ARENA_SIZE, ARENA_SIZE),
                     HOVER_HEIGHT])

def nearest_node(tree, config):
    distances = [np.linalg.norm(node.pos - config) for node in tree]
    return tree[np.argmin(distances)]

def steer(from_pos, to_pos):
    direction = to_pos - from_pos
    distance = np.linalg.norm(direction)
    if distance > DELTA:
        direction = (direction / distance) * DELTA
    return from_pos + direction

def is_collision_free(drone, from_pos, to_pos):
    direction = to_pos - from_pos
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return True
    unit_direction = direction / distance
    steps = int(distance / COLLISION_CHECK_STEP)
    for i in range(steps + 1):
        check_pos = from_pos + unit_direction * COLLISION_CHECK_STEP * i
        if drone.is_collision(check_pos):
            return False
    return True

def reconstruct_path(goal_node):
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node.pos)
        current_node = current_node.parent
    return path[::-1]

# --- Create Results Directory ---
RESULTS_DIR = "rrt_simulation_results_csv"
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
    results_file = os.path.join(RESULTS_DIR, f"rrt_num_drones_{num_drones}.csv")
    print(f"Running RRT with {num_drones} drones...")

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

            # Generate random start and target positions
            unknown_target_pos = generate_random_target_pos()

            # Initialize drones and RRTs
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
                for drone in drones:
                    # RRT algorithm for each drone
                    if not drone.target_found:
                        random_config = get_random_config()
                        nearest = nearest_node(drone.tree, random_config)
                        new_pos = steer(nearest.pos, random_config)

                        if is_collision_free(drone, nearest.pos, new_pos):
                            new_node = Node(new_pos)
                            new_node.parent = nearest
                            new_node.signal_strength = get_signal_strength(new_pos, unknown_target_pos)
                            drone.tree.append(new_node)

                            # Check if target is found based on signal strength
                            if new_node.signal_strength > SIGNAL_FOUND_THRESHOLD:
                                print(f"    Target found by drone {drone.id} in iteration {iteration+1}!")
                                drone.target_found = True
                                drone.target_found_in_iteration = iteration + 1
                                if not target_found: # Update global target found only once
                                    target_found = True
                                    iterations_to_find_target = iteration + 1

                                path = reconstruct_path(new_node)
                                drone.path = path

                    # Move drone along the path
                    if drone.target_found:
                        drone.move_along_path()

                # Simulation step
                p.stepSimulation()

                if all(drone.target_found for drone in drones):
                    break

            # Write results to file
            result = {
                "run": run + 1,
                "num_drones": num_drones,
                "target_found": target_found,
                "iterations": iterations_to_find_target if target_found else NUM_ITERATIONS,
                "target_pos": list(unknown_target_pos)
            }
            for i, drone in enumerate(drones):
                result[f"drone_{i}_start_pos"] = list(start_positions[i%len(start_positions)])
                result[f"drone_{i}_found_in_iteration"] = drone.target_found_in_iteration if drone.target_found else -1
            writer.writerow(result)

# --- End Simulation ---
p.disconnect()