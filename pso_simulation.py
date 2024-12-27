import gymnasium as gym
import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import os

# --- Parameters ---
NUM_DRONES = 20
NUM_ITERATIONS = 2000
C1 = 2.0  # Cognitive parameter (influence of personal best)
C2 = 2.0  # Social parameter (influence of global best)
W = 0.7  # Inertia weight
V_MAX = 0.1  # Maximum velocity
DRONE_RADIUS = 0.1 # Radius of the drone
HOVER_HEIGHT = 1.0
ARENA_SIZE = 5
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

# --- Define Goal (Target) ---
target_pos = [4, 4, HOVER_HEIGHT]

# Visualize the target
target_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 1, 0, 1])
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
        self.pos = np.array(start_pos, dtype=np.float64)
        self.velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity
        self.best_pos = self.pos.copy()  # Personal best position
        self.best_fitness = float('-inf')  # Best fitness achieved by this drone
        self.body = p.loadURDF(quadrotor_urdf_path, self.pos)
        self.target_found = False
        self.target_found_in_iteration = -1
        
    def calculate_fitness(self):
        # Fitness is the inverse of the distance to the target.
        distance_to_target = np.linalg.norm(self.pos - np.array(target_pos))
        fitness = 1.0 / (distance_to_target + 1e-6)
        return fitness

    def update_personal_best(self):
        fitness = self.calculate_fitness()
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_pos = self.pos.copy()

    def update_velocity(self, global_best_pos):
        # Update velocity based on personal best, global best, and inertia
        r1 = np.random.rand(3)
        r2 = np.random.rand(3)

        cognitive_component = C1 * r1 * (self.best_pos - self.pos)
        social_component = C2 * r2 * (global_best_pos - self.pos)
        self.velocity = W * self.velocity + cognitive_component + social_component

        # Limit velocity
        self.velocity = np.clip(self.velocity, -V_MAX, V_MAX)

    def move(self):
        self.pos += self.velocity
        
        # Check if target is reached
        if np.linalg.norm(self.pos - np.array(target_pos)) < TARGET_REACH_THRESHOLD:
            self.target_found = True
            self.target_found_in_iteration = iteration
            self.pos = np.array(target_pos) # If the target is reached, move drone to the exact target position

        # Check for collisions
        if self.is_collision():
            self.pos -= self.velocity # Move drone back to previous position if collision
            self.velocity = -0.5 * self.velocity  # Reflect velocity

        # Keep the drone within the arena bounds
        self.pos[0] = np.clip(self.pos[0], -ARENA_SIZE, ARENA_SIZE)
        self.pos[1] = np.clip(self.pos[1], -ARENA_SIZE, ARENA_SIZE)
        self.pos[2] = np.clip(self.pos[2], 0, ARENA_SIZE) # Keep the drone above the ground

        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))
    
    def is_collision(self):
        # Check for collisions with obstacles
        for obstacle_id in obstacles:
            closest_points = p.getClosestPoints(self.body, obstacle_id, distance=DRONE_RADIUS*2)
            if closest_points:
                return True
        return False

# --- Initialize Drones ---
drones = [Drone(i, [random.uniform(-ARENA_SIZE, ARENA_SIZE), random.uniform(-ARENA_SIZE, ARENA_SIZE), HOVER_HEIGHT]) for i in range(NUM_DRONES)]

# --- Main Loop ---
global_best_pos = drones[0].pos.copy()
global_best_fitness = float('-inf')
target_found = False
iterations_to_find_target = 0

for iteration in range(NUM_ITERATIONS):
    print(f"Iteration: {iteration + 1}")

    # Update personal and global bests
    for drone in drones:
        drone.update_personal_best()
        if drone.best_fitness > global_best_fitness:
            global_best_fitness = drone.best_fitness
            global_best_pos = drone.best_pos.copy()

    # Update velocity and move drones
    for drone in drones:
        drone.update_velocity(global_best_pos)
        drone.move()
    # Check if any drone has found the target
    
    for drone in drones:
        if drone.target_found and drone.target_found_in_iteration != -1:
            target_found = True
            iterations_to_find_target = drone.target_found_in_iteration + 1
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