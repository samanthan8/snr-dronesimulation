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
EVAPORATION_RATE = 0.5
ALPHA = 1  # Importance of pheromone
BETA = 2   # Importance of heuristic (distance)
Q = 100  # Pheromone deposit amount
INITIAL_PHEROMONE = 1.0
DRONE_SPEED = 0.05  # Max speed of the drones
SEARCH_RADIUS = 1.0  # Radius for collision checks
GRID_SIZE = 0.5 # Pheromone grid size
HOVER_HEIGHT = 1.0 # Height at which drones hover
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

# --- Define Start and Goal ---
start_pos = [-4, -4, HOVER_HEIGHT]
goal_pos = [4, 4, HOVER_HEIGHT]

# Visualize start and goal
p.addUserDebugLine(start_pos, [start_pos[0], start_pos[1], start_pos[2] + 0.5], [1, 0, 0], lineWidth=3)
p.addUserDebugLine(goal_pos, [goal_pos[0], goal_pos[1], goal_pos[2] + 0.5], [0, 1, 0], lineWidth=3)

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

# --- Pheromone Grid ---
grid_x_min, grid_x_max = -ARENA_SIZE, ARENA_SIZE
grid_y_min, grid_y_max = -ARENA_SIZE, ARENA_SIZE
grid_z_min, grid_z_max = 0, 3

pheromone_grid = {}
pheromone_line_ids = {}  # Dictionary to store line IDs

# Offset for ID generation to avoid negative IDs
id_offset = ARENA_SIZE

for x in np.arange(grid_x_min, grid_x_max + GRID_SIZE, GRID_SIZE):
    for y in np.arange(grid_y_min, grid_y_max + GRID_SIZE, GRID_SIZE):
        for z in np.arange(grid_z_min, grid_z_max + GRID_SIZE, GRID_SIZE):
            rounded_x, rounded_y, rounded_z = round(x, 1), round(y, 1), round(z, 1)
            pheromone_grid[(rounded_x, rounded_y, rounded_z)] = INITIAL_PHEROMONE

            # Visualize the pheromone grid - only at HOVER_HEIGHT for simplicity
            if rounded_z == HOVER_HEIGHT:
                # Calculate unique ID with offset
                unique_id = int((rounded_x + id_offset) * 1000 + (rounded_y + id_offset) * 10 + (rounded_z + id_offset))
                line_id = p.addUserDebugLine([x, y, z], [x, y, z + 0.05], [0, 0, pheromone_grid[(rounded_x, rounded_y, rounded_z)] / INITIAL_PHEROMONE],
                                          lineWidth=pheromone_grid[(rounded_x, rounded_y, rounded_z)] / INITIAL_PHEROMONE)
                pheromone_line_ids[(rounded_x, rounded_y, rounded_z)] = line_id

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
        self.target_found_in_iteration = -1
        self.body = p.loadURDF(quadrotor_urdf_path, self.pos)

    def calculate_probabilities(self):
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
                    distance_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(next_cell))
                    heuristic = 1.0 / (distance_to_goal + 1e-6)

                    attractiveness = (pheromone ** ALPHA) * (heuristic ** BETA)
                    probabilities[next_cell] = attractiveness
                    total += attractiveness

        # Normalize probabilities
        for cell in probabilities:
            probabilities[cell] /= total

        return probabilities

    def choose_next_cell(self):
        probabilities = self.calculate_probabilities()
        if not probabilities:
            return get_cell_from_pos(self.pos)

        rand_num = random.random()
        cumulative_prob = 0
        for cell, prob in probabilities.items():
            cumulative_prob += prob
            if rand_num <= cumulative_prob:
                return cell

    def move(self):
        if self.has_found_goal:
            if len(self.path_positions) > 1:
                next_pos = self.path_positions[-2]
                self.path_positions.pop()
                self.pos = self.calculate_movement(next_pos) # Use smooth movement
            else:
                self.pos = self.calculate_movement(self.start_pos) # Use smooth movement
                self.path = [get_cell_from_pos(self.start_pos)]
                self.path_positions = [self.start_pos]
                self.has_found_goal = False
        else:
            next_cell = self.choose_next_cell()
            target_pos = [next_cell[0], next_cell[1], next_cell[2]]

            if np.linalg.norm(np.array(self.pos) - np.array(goal_pos)) < GRID_SIZE:
                self.has_found_goal = True
                self.target_found_in_iteration = iteration                
            else:
                self.path.append(next_cell)
                self.path_positions.append(target_pos)
                self.pos = self.calculate_movement(target_pos) # Use smooth movement

        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0]))

    def calculate_movement(self, target_pos):
        """Calculates a movement vector towards the target, considering speed."""
        direction = np.array(target_pos) - np.array(self.pos)
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance

        # Limit the movement to the drone's speed
        actual_movement = min(distance, DRONE_SPEED)
        new_pos = np.array(self.pos) + direction * actual_movement

        return list(new_pos)

    def is_collision(self, next_pos):
        # Check for collisions with obstacles
        p.resetBasePositionAndOrientation(self.body, next_pos, p.getQuaternionFromEuler([0, 0, 0])) # Temporarily move drone to check for collision
        for obstacle_id in obstacles:
            closest_points = p.getClosestPoints(self.body, obstacle_id, distance=SEARCH_RADIUS)
            if closest_points:
                p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0])) # Reset drone position after collision check
                return True
        p.resetBasePositionAndOrientation(self.body, self.pos, p.getQuaternionFromEuler([0, 0, 0])) # Reset drone position after collision check
        return False

    def deposit_pheromone(self):
        if self.has_found_goal:
            path_length = len(self.path)
            pheromone_amount = Q / path_length
            for cell in self.path:
                pheromone_grid[cell] += pheromone_amount

# --- Initialize Drones ---
drones = [Drone(i, start_pos) for i in range(NUM_DRONES)]

# --- Main Loop ---
target_found = False
iterations_to_find_target = 0

for iteration in range(NUM_ITERATIONS):
    print(f"Iteration: {iteration + 1}")

    # Move drones
    for drone in drones:
        drone.move()

    # Deposit pheromone
    for drone in drones:
        drone.deposit_pheromone()

    # Evaporate pheromone
    for cell in pheromone_grid:
        pheromone_grid[cell] *= (1 - EVAPORATION_RATE)

        # Check for NaN or inf
        if math.isnan(pheromone_grid[cell]) or math.isinf(pheromone_grid[cell]):
            print(f"Invalid pheromone value in cell: {cell}, Value: {pheromone_grid[cell]}")
            pheromone_grid[cell] = INITIAL_PHEROMONE  # Reset to a default value

        # Update pheromone visualization (only at HOVER_HEIGHT for simplicity)
        if cell[2] == HOVER_HEIGHT and cell in pheromone_line_ids:
            pheromone_level = max(0, min(pheromone_grid[cell], 5))
            color = [0, 0, pheromone_level / 5]

            # Use stored line ID for updating
            p.addUserDebugLine([cell[0], cell[1], cell[2]], [cell[0], cell[1], cell[2] + 0.05], color,
                              lineWidth=pheromone_level / 2, replaceItemUniqueId=pheromone_line_ids[cell])
    
    # Check if any drone has found the target
    for drone in drones:
        if drone.has_found_goal and drone.target_found_in_iteration != -1:
            target_found = True
            iterations_to_find_target = drone.target_found_in_iteration + 1
            break
        
    # Simulation step
    p.stepSimulation()
    time.sleep(1./240.)

# --- End Simulation ---
p.disconnect()

# Print the result
if target_found:
    print(f"Target found in {iterations_to_find_target} iterations!")
else:
    print("Target not found within the maximum number of iterations.")