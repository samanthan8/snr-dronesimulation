import gymnasium as gym
import pybullet as p
import pybullet_data
import time
import numpy as np
import random

# --- Parameters ---
NUM_DRONES = 5
NUM_ITERATIONS = 5000
DELTA = 0.5
COLLISION_CHECK_STEP = 0.05
DRONE_RADIUS = 0.1
HOVER_HEIGHT = 1.0
ARENA_SIZE = 5
SIGNAL_FOUND_THRESHOLD = 0.7
SEARCH_RADIUS = 0.5 # Radius for collision detection

# --- Environment Setup ---
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# --- Start and Target Positions ---
start_positions = [[-4, -4, HOVER_HEIGHT],
                   [-4, 4, HOVER_HEIGHT],
                   [4, -4, HOVER_HEIGHT],
                   [0, 4, HOVER_HEIGHT],
                   [0, 0, HOVER_HEIGHT]]  # Example: Multiple start positions

# The target's position is unknown to the drone initially.
unknown_target_pos = [4, 4, HOVER_HEIGHT]

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

# --- Signal Strength Function ---
def get_signal_strength(drone_pos, target_pos):
    distance = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
    if distance < 0.1:
        return 1.0
    else:
        return 1.0 / (distance ** 2)

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
        self.body = p.loadURDF("quadrotor.urdf", start_pos)
        self.path = []
        self.target_found = False
        self.target_found_in_iteration = -1
        self.tree = [Node(start_pos)] # Each drone maintains its own RRT

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

# --- Initialize Drones ---
drones = [Drone(i, start_positions[i%len(start_positions)]) for i in range(NUM_DRONES)]

# --- Main Loop ---
target_found = False
iterations_to_find_target = 0
for iteration in range(NUM_ITERATIONS):
    print(f"Iteration: {iteration + 1}")

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

                # Draw the new edge
                p.addUserDebugLine(nearest.pos, new_pos, [0, 0, 1], lineWidth=2)

                # Check if target is found based on signal strength
                if new_node.signal_strength > SIGNAL_FOUND_THRESHOLD:
                    print(f"Target found by drone {drone.id} in iteration {iteration+1}!")
                    drone.target_found = True
                    drone.target_found_in_iteration = iteration + 1
                    if not target_found: # Update global target found only once
                        target_found = True
                        iterations_to_find_target = iteration + 1

                    path = reconstruct_path(new_node)
                    drone.path = path
                    # Visualize the final path
                    for i in range(len(path) - 1):
                        p.addUserDebugLine(path[i], path[i+1], [1, 0, 0], lineWidth=3, lifeTime=0)

        # Move drone along the path
        if drone.target_found:
            drone.move_along_path()

    # Simulation step
    p.stepSimulation()
    time.sleep(1./240.)

    if all(drone.target_found for drone in drones):
        break

# --- End Simulation ---
p.disconnect()

# Print the result
if target_found:
    print(f"Target found in {iterations_to_find_target} iterations!")
else:
    print("Target not found within the maximum number of iterations.")