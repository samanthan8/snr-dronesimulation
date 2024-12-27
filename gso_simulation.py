import gymnasium as gym
import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import math
import os

# --- Parameters ---
NUM_DRONES = 20
NUM_ITERATIONS = 2000
INITIAL_LUCIFERIN = 5.0  # Initial luciferin value for each drone
LUCIFERIN_DECAY_RATE = 0.4  # Decay rate of luciferin
LUCIFERIN_ENHANCE_RATE = 0.6 # Rate of enhancing luciferin based on objective function
GAMMA = 0.6 # Parameter for updating the neighborhood range (sensor range)
BETA = 0.08 # Constant for variation in the neighborhood range (sensor range)
S = 0.03 # Step size
RS = 2.0 # Initial sensor range
DRONE_SPEED = 0.05 # Maximum speed of the drones
HOVER_HEIGHT = 1.0 # Height at which drones hover
ARENA_SIZE = 5
MAX_NEIGHBORS = 5 # Maximum number of neighbors to consider
NT = 5 # Desired number of neighbors
TARGET_REACH_THRESHOLD = 0.5

# Define the path to the local directory where the URDF files are stored
LOCAL_URDF_PATH = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
# Define the URDF file paths
plane_urdf_path = os.path.join(LOCAL_URDF_PATH, "plane.urdf")  # Make sure 'plane.urdf' is in the same folder
quadrotor_urdf_path = os.path.join(LOCAL_URDF_PATH, "quadrotor.urdf")  # Make sure 'quadrotor.urdf' is in the same folder

# --- Environment Setup ---
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF(plane_urdf_path)

# --- Define Start and Goal (Target) ---
# We'll use the goal as the "target" that emits the "light" the glowworms seek
target_pos = [4, 4, HOVER_HEIGHT]

# Visualize the target
target_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 1, 0, 1])  # Yellow sphere
target_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_shape, basePosition=target_pos)

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

# --- Drone Class ---
class Drone:
    def __init__(self, id, start_pos):
        self.id = id
        self.pos = start_pos
        self.luciferin = INITIAL_LUCIFERIN
        self.sensor_range = RS
        self.body = p.loadURDF(quadrotor_urdf_path, self.pos)
        self.target_found = False
        
    def calculate_objective_function(self):
        # The objective function is the inverse of the distance to the target.
        # We want to maximize this function (closer to the target is better).
        distance_to_target = np.linalg.norm(np.array(self.pos) - np.array(target_pos))
        objective = 1.0 / (distance_to_target + 1e-6) # Added a small value to avoid division by zero
        return objective

    def update_luciferin(self):
        # Update luciferin based on the objective function
        self.luciferin = (1 - LUCIFERIN_DECAY_RATE) * self.luciferin + LUCIFERIN_ENHANCE_RATE * self.calculate_objective_function()
        

    def find_neighbors(self, drones):
        neighbors = []
        for other in drones:
            if other.id != self.id:
                dist = np.linalg.norm(np.array(self.pos) - np.array(other.pos))
                if dist <= self.sensor_range and self.luciferin < other.luciferin:
                    neighbors.append((other, dist))
        # Sort neighbors by distance (ascending)
        neighbors.sort(key=lambda x: x[1])

        # Limit the number of neighbors
        neighbors = neighbors[:MAX_NEIGHBORS]

        return [n[0] for n in neighbors]

    def move_towards_neighbors(self, neighbors):
        if not neighbors:
            return
        
        # Calculate the weighted average direction towards neighbors
        total_weight = 0
        weighted_direction = np.array([0.0, 0.0, 0.0])

        for neighbor in neighbors:
            direction = np.array(neighbor.pos) - np.array(self.pos)
            distance = np.linalg.norm(direction)
            
            # Avoid division by zero if the drone is too close to a neighbor
            if distance > 1e-6:
                weight = (neighbor.luciferin - self.luciferin) / distance
                weighted_direction += weight * direction
                total_weight += weight
        
        # Normalize the weighted direction
        if total_weight > 0:
            weighted_direction /= total_weight

        # Move the drone towards the neighbors
        new_pos = np.array(self.pos) + S * weighted_direction
        self.pos = list(new_pos)
        
        # Check if target is reached
        if np.linalg.norm(new_pos - np.array(target_pos)) < TARGET_REACH_THRESHOLD:
            self.target_found = True
            new_pos = np.array(target_pos) # If the target is reached, move drone to the exact target position        

        # Simple collision avoidance: If a collision is detected, move back a bit
        if self.is_collision(self.pos):
            self.pos = list(np.array(self.pos) - 0.5 * S * weighted_direction)

        # Keep the drone within the arena bounds
        self.pos[0] = np.clip(self.pos[0], -ARENA_SIZE, ARENA_SIZE)
        self.pos[1] = np.clip(self.pos[1], -ARENA_SIZE, ARENA_SIZE)
        self.pos[2] = np.clip(self.pos[2], 0, ARENA_SIZE)

        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))

    def update_sensor_range(self, neighbors):
        # Update the sensor range based on the number of neighbors
        self.sensor_range = self.sensor_range + BETA * (NT - len(neighbors))

        # Keep the sensor range within bounds [0, RS]
        self.sensor_range = max(0, min(self.sensor_range, RS))

    def is_collision(self, next_pos):
        # Check for collisions with obstacles
        p.resetBasePositionAndOrientation(self.body, next_pos, p.getQuaternionFromEuler([0, 0, 0]))
        for obstacle_id in obstacles:
            closest_points = p.getClosestPoints(self.body, obstacle_id, distance=0.5)
            if closest_points:
                p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))
                return True
        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))
        return False

# --- Initialize Drones ---
drones = [Drone(i, [random.uniform(-ARENA_SIZE, ARENA_SIZE), random.uniform(-ARENA_SIZE, ARENA_SIZE), HOVER_HEIGHT]) for i in range(NUM_DRONES)]

# --- Main Loop ---
target_found = False
iterations_to_find_target = 0

for iteration in range(NUM_ITERATIONS):
    print(f"Iteration: {iteration + 1}")

    # Update luciferin for each drone
    for drone in drones:
        drone.update_luciferin()

    # Update sensor range and find neighbors for each drone
    for drone in drones:
        neighbors = drone.find_neighbors(drones)
        drone.update_sensor_range(neighbors)

    # Move drones towards neighbors
    for drone in drones:
        neighbors = drone.find_neighbors(drones)
        drone.move_towards_neighbors(neighbors)
        
    for drone in drones:
        if drone.target_found:
            target_found = True
            iterations_to_find_target = iteration + 1
            break    

    # Simulation step
    p.stepSimulation()
    time.sleep(1./240.)
    
    # Exit the loop if the target is found
    if target_found:
        break
    
# --- End Simulation ---
p.disconnect()

# Print the result
if target_found:
    print(f"Target found in {iterations_to_find_target} iterations!")
else:
    print("Target not found within the maximum number of iterations.")