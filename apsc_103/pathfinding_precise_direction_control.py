import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from itertools import permutations
from math import sqrt, radians, cos, sin

# Grid and robot parameters
GRID_WIDTH = 50
GRID_HEIGHT = 25
OBSTACLES = [(28, 17), (13, 13)]
OBSTACLE_SIZE = 4
ROBOT_RADIUS = 2
START_POINT = (1, 1)
CHECKPOINTS = [(13, 18), (2, 10),(16,23)]

# Generate precise movement directions with exact costs
def generate_precise_directions():
    # Basic 8 directions
    directions = {
        (0, 1): 1, (0, -1): 1, (1, 0): 1, (-1, 0): 1,  # Cardinal
        (1, 1): sqrt(2), (1, -1): sqrt(2), (-1, 1): sqrt(2), (-1, -1): sqrt(2)  # Diagonal
    }
    
    # Add 11.25° directions
    angle_increment = 11.25
    for angle_deg in np.arange(0, 360, angle_increment):
        if angle_deg % 45 == 0:  # Skip angles already covered by basic 8 directions
            continue
            
        angle_rad = radians(angle_deg)
        # Calculate exact direction vectors
        dx, dy = cos(angle_rad), sin(angle_rad)
        
        # Find closest grid point within max radius
        max_radius = 5
        best_grid_point = None
        min_error = float('inf')
        
        for r in range(1, max_radius + 1):
            grid_x = round(dx * r)
            grid_y = round(dy * r)
            
            if grid_x == 0 and grid_y == 0:
                continue
                
            # Skip if this is effectively one of our basic 8 directions
            if (grid_x, grid_y) in directions or (-grid_x, -grid_y) in directions:
                continue
                
            # Calculate how close this grid point is to the true angle
            true_angle = np.arctan2(grid_y, grid_x)
            angle_error = abs(true_angle - angle_rad)
            
            if angle_error < min_error:
                min_error = angle_error
                best_grid_point = (grid_x, grid_y)
        
        if best_grid_point:
            x, y = best_grid_point
            # Use exact Euclidean distance for the cost
            directions[(x, y)] = sqrt(x*x + y*y)
    
    return directions

MOVE_COSTS = generate_precise_directions()

def expand_obstacles(obstacles, grid_width, grid_height):
    blocked_cells = set()
    buffer = OBSTACLE_SIZE // 2 + ROBOT_RADIUS
    for ox, oy in obstacles:
        for dx in range(-buffer, buffer + 1):
            for dy in range(-buffer, buffer + 1):
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    blocked_cells.add((nx, ny))
    return blocked_cells

BLOCKED_CELLS = expand_obstacles(OBSTACLES, GRID_WIDTH, GRID_HEIGHT)

def get_move_direction(from_pos, to_pos):
    """Get the movement direction from one position to another, if it exists in MOVE_COSTS"""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    if (dx, dy) in MOVE_COSTS:
        return (dx, dy)
    
    return None

def is_valid_move(from_pos, to_pos):
    """Check if a move is valid (exists in movement directions)"""
    direction = get_move_direction(from_pos, to_pos)
    return direction is not None

def dijkstra(start, goal, blocked_cells, grid_width, grid_height, max_iterations=100000):
    heap = [(0, start, [])]
    min_cost = {start: 0}
    visited = set()
    iterations = 0

    while heap and iterations < max_iterations:
        iterations += 1
        cost, current, path = heappop(heap)
        
        if current == goal:
            return path + [current]
            
        if current in visited:
            continue
            
        visited.add(current)

        for (dx, dy), move_cost in MOVE_COSTS.items():
            nx, ny = current[0] + dx, current[1] + dy
            next_pos = (nx, ny)
            
            # Check bounds
            if not (0 <= nx < grid_width and 0 <= ny < grid_height):
                continue
                
            # Check for collisions - integrated collision detection
            if next_pos in blocked_cells:
                continue
                
            # Line-of-sight collision check for longer moves
            if max(abs(dx), abs(dy)) > 1:
                has_collision = False
                # Simple Bresenham-like line check
                steps = max(abs(dx), abs(dy))
                for step in range(1, steps):
                    check_x = int(current[0] + dx * step / steps)
                    check_y = int(current[1] + dy * step / steps)
                    if (check_x, check_y) in blocked_cells:
                        has_collision = True
                        break
                if has_collision:
                    continue
            
            new_cost = cost + move_cost
            if next_pos not in min_cost or new_cost < min_cost[next_pos]:
                min_cost[next_pos] = new_cost
                heappush(heap, (new_cost, next_pos, path + [current]))

    print(f"Warning: Dijkstra search terminated after {iterations} iterations without finding a path")
    return None

def precompute_paths(start, checkpoints):
    points = [start] + checkpoints
    path_dict = {}
    
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                print(f"Finding path from {p1} to {p2}...")
                path = dijkstra(p1, p2, BLOCKED_CELLS, GRID_WIDTH, GRID_HEIGHT)
                
                if path:
                    # Validate the path
                    is_valid = True
                    for k in range(1, len(path)):
                        if not is_valid_move(path[k-1], path[k]):
                            print(f"Warning: Invalid move in path from {p1} to {p2}: {path[k-1]} -> {path[k]}")
                            is_valid = False
                            break
                    
                    if is_valid:
                        path_dict[(p1, p2)] = path
                    else:
                        print(f"Path from {p1} to {p2} contains invalid moves. Discarding.")
                else:
                    print(f"No path found from {p1} to {p2}")
    
    return path_dict

def calculate_path_cost(path):
    """Calculate the true cost of a path using the MOVE_COSTS dictionary"""
    if not path or len(path) < 2:
        return 0
        
    cost = 0
    for i in range(1, len(path)):
        from_pos = path[i-1]
        to_pos = path[i]
        direction = get_move_direction(from_pos, to_pos)
        
        if direction is None:
            print(f"Warning: No valid direction for move {from_pos} -> {to_pos}")
            # Fallback to Euclidean distance if direction not found
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            cost += sqrt(dx*dx + dy*dy)
        else:
            cost += MOVE_COSTS[direction]
            
    return cost

def find_optimal_checkpoint_path(start, checkpoints, path_dict):
    best_path = None
    min_total_cost = float('inf')
    
    # Tracking which permutations lead to complete paths
    valid_perms = 0
    total_perms = 0
    
    for perm in permutations(checkpoints):
        total_perms += 1
        full_path = []
        total_cost = 0
        current = start
        valid_sequence = True
        
        for cp in perm:
            if (current, cp) in path_dict:
                segment = path_dict[(current, cp)]
                # Skip the first point of the segment (except for first segment)
                if full_path:
                    segment_to_add = segment[1:]
                else:
                    segment_to_add = segment
                
                full_path.extend(segment_to_add)
                
                # Use correct cost calculation
                segment_cost = calculate_path_cost(segment)
                total_cost += segment_cost
                
                current = cp
            else:
                valid_sequence = False
                break
                
        if valid_sequence:
            valid_perms += 1
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_path = full_path
    
    print(f"Evaluated {total_perms} permutations, {valid_perms} were valid.")
    
    if best_path is None:
        print("❌ No complete path found to all checkpoints.")
    else:
        print(f"✅ Found optimal path with cost: {min_total_cost:.3f}")
        
    return best_path

def validate_path(path):
    """Comprehensive path validation"""
    if not path or len(path) < 2:
        return True
        
    for i in range(1, len(path)):
        from_pos = path[i-1]
        to_pos = path[i]
        
        # Check if the move is in our movement set
        if not is_valid_move(from_pos, to_pos):
            print(f"Invalid move: {from_pos} -> {to_pos}")
            return False
            
        # Check for collision
        if to_pos in BLOCKED_CELLS:
            print(f"Collision detected at {to_pos}")
            return False
            
        # Check line-of-sight for longer moves
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if max(abs(dx), abs(dy)) > 1:
            # Simple Bresenham-like line check
            steps = max(abs(dx), abs(dy))
            for step in range(1, steps):
                check_x = int(from_pos[0] + dx * step / steps)
                check_y = int(from_pos[1] + dy * step / steps)
                if (check_x, check_y) in BLOCKED_CELLS:
                    print(f"Line-of-sight collision at {(check_x, check_y)}")
                    return False
    
    return True

# Run full logic
print("Precomputing paths between points...")
precomputed_paths = precompute_paths(START_POINT, CHECKPOINTS)

print("\nFinding optimal checkpoint order...")
optimal_path = find_optimal_checkpoint_path(START_POINT, CHECKPOINTS, precomputed_paths)

if optimal_path:
    full_path = [START_POINT] + (optimal_path[1:] if optimal_path[0] == START_POINT else optimal_path)
    
    print("\nValidating final path...")
    if validate_path(full_path):
        print("✅ Path sequence is valid.")
        print("🔢 Total steps:", len(optimal_path))
        print("📏 Total path cost:", round(calculate_path_cost(full_path), 3))
    else:
        print("❌ Path contains invalid steps or collisions.")
else:
    print("❌ No valid path exists through all checkpoints.")
    full_path = [START_POINT]

# Visualization with improved details
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, GRID_WIDTH)
ax.set_ylim(0, GRID_HEIGHT)
ax.set_xticks(np.arange(0, GRID_WIDTH + 1, 5))
ax.set_yticks(np.arange(0, GRID_HEIGHT + 1, 5))
plt.grid(True, alpha=0.3)

# Plot movement vectors for reference
if False:  # Set to True to visualize all allowed moves
    for (dx, dy), cost in MOVE_COSTS.items():
        center = (GRID_WIDTH//2, GRID_HEIGHT//2)
        end = (center[0] + dx, center[1] + dy)
        ax.arrow(center[0], center[1], dx, dy, head_width=0.3, 
                 head_length=0.3, fc='gray', ec='gray', alpha=0.5)

# Draw obstacles
for ox, oy in OBSTACLES:
    buffer = OBSTACLE_SIZE // 2 + ROBOT_RADIUS
    obstacle_rect = plt.Rectangle((ox-buffer, oy-buffer), 
                                  buffer*2, buffer*2, 
                                  color="red", alpha=0.3)
    ax.add_patch(obstacle_rect)
    # Draw the actual physical obstacle
    physical_rect = plt.Rectangle((ox-OBSTACLE_SIZE//2, oy-OBSTACLE_SIZE//2),
                                 OBSTACLE_SIZE, OBSTACLE_SIZE,
                                 color="red", alpha=0.7)
    ax.add_patch(physical_rect)

# Draw path with direction indicators
if len(full_path) > 1:
    # Plot the path
    x_coords, y_coords = zip(*full_path)
    ax.plot(x_coords, y_coords, marker="o", color="blue", 
            linestyle="-", label="Path", markersize=3)
    
    # Highlight checkpoints in the order they're visited
    checkpoint_indices = []
    for i, cp in enumerate(CHECKPOINTS):
        # Find where this checkpoint appears in the path
        if cp in full_path:
            checkpoint_indices.append((full_path.index(cp), i))
    
    # Sort by order of appearance in path
    checkpoint_indices.sort()
    
    for path_idx, cp_idx in checkpoint_indices:
        cp = full_path[path_idx]
        ax.text(cp[0]+0.3, cp[1]+0.3, f"{checkpoint_indices.index((path_idx, cp_idx))+1}", 
                fontsize=12, fontweight='bold')

# Start and checkpoints
ax.scatter(*START_POINT, color="green", s=120, label="Start", zorder=10)
for i, cp in enumerate(CHECKPOINTS):
    ax.scatter(*cp, color="orange", s=160, edgecolor="black", linewidth=1.5,
               label="Checkpoint" if i == 0 else "", zorder=9)

ax.set_xlabel("X [units]")
ax.set_ylabel("Y [units]")
ax.set_title("Improved Pathfinding with Precise Direction Control")
ax.legend()
plt.tight_layout()
plt.show()