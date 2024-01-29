import numpy as np
import matplotlib.pyplot as plt

# Grid cell properties
class GridCell:
    def __init__(self, phase, orientation, scale):
        self.phase = phase  # Phase offset
        self.orientation = orientation  # Orientation in radians
        self.scale = scale  # Spatial scale of the grid pattern

# Environment properties
class Environment:
    def __init__(self, size):
        self.size = size  # Size of the environment (e.g., 100x100 units)

# Initialize grid cells and environment
def initialize_grid_cells(num_cells, env_size):
    np.random.seed(42)  # For reproducibility
    cells = []
    for _ in range(num_cells):
        phase = np.random.rand() * 2 * np.pi
        orientation = np.random.rand() * 2 * np.pi
        scale = np.random.uniform(0.5, 2.0)  # Arbitrary scale values
        cells.append(GridCell(phase, orientation, scale))
    env = Environment(env_size)
    return cells, env

# Initialize cells and environment
num_cells = 10  # Number of grid cells
env_size = (100, 100)  # Environment size (can be adjusted)
grid_cells, environment = initialize_grid_cells(num_cells, env_size)




def simulate_movement(grid_cells, environment, steps=1000):
    # Simulating random movement in the environment
    x, y = np.random.randint(0, environment.size[0]), np.random.randint(0, environment.size[1])
    trajectory = np.zeros((steps, 2))  # Store (x, y) positions
    cell_responses = np.zeros((len(grid_cells), steps))  # Store cell responses

    for t in range(steps):
        # Random walk (this can be modified for different movement patterns)
        dx, dy = np.random.randint(-1, 2), np.random.randint(-1, 2)
        x = np.clip(x + dx, 0, environment.size[0] - 1)
        y = np.clip(y + dy, 0, environment.size[1] - 1)
        trajectory[t] = [x, y]

        # Compute response of each grid cell
        for i, cell in enumerate(grid_cells):
            # Simple model of grid cell response
            response = np.cos((x * np.cos(cell.orientation) + y * np.sin(cell.orientation)) / cell.scale + cell.phase)
            cell_responses[i, t] = response

    return trajectory, cell_responses

# Simulate movement and get cell responses
trajectory, cell_responses = simulate_movement(grid_cells, environment)


def plot_place_fields(grid_cells, trajectory, cell_responses, environment):
    plt.figure(figsize=(10, 10))
    for i, cell in enumerate(grid_cells):
        plt.subplot(5, 2, i+1)  # Adjust subplot layout based on the number of cells
        # Compute average response for each position
        place_field = np.zeros(environment.size)
        for t in range(len(trajectory)):
            x, y = int(trajectory[t, 0]), int(trajectory[t, 1])
            place_field[x, y] += cell_responses[i, t]
        place_field /= len(trajectory)
        
        plt.imshow(place_field, cmap='hot', interpolation='nearest')
        plt.title(f'Cell {i+1}')
        plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plotting the place fields
#plot_place_fields(grid_cells, trajectory, cell_responses, environment)

def improved_movement_simulation(grid_cells, environment, steps=1000):
    x, y = np.random.randint(0, environment.size[0]), np.random.randint(0, environment.size[1])
    trajectory = np.zeros((steps, 2))
    cell_responses = np.zeros((len(grid_cells), steps))

    direction = np.random.rand() * 2 * np.pi  # Initial random direction

    for t in range(steps):
        # Add some directional persistence
        direction += np.random.randn() * 0.1  # Small random change in direction
        dx, dy = np.cos(direction), np.sin(direction)
        x = np.clip(x + dx, 0, environment.size[0] - 1)
        y = np.clip(y + dy, 0, environment.size[1] - 1)
        trajectory[t] = [x, y]

        for i, cell in enumerate(grid_cells):
            response = np.cos((x * np.cos(cell.orientation) + y * np.sin(cell.orientation)) / cell.scale + cell.phase)
            cell_responses[i, t] = response

    return trajectory, cell_responses

# Improved movement simulation
trajectory, cell_responses = improved_movement_simulation(grid_cells, environment)
plot_place_fields(grid_cells, trajectory, cell_responses, environment)

def analyze_cell_responses(cell_responses):
    correlation_matrix = np.corrcoef(cell_responses)
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title("Correlation Matrix of Grid Cell Responses")
    plt.xlabel("Grid Cell")
    plt.ylabel("Grid Cell")
    plt.show()

# Analyzing the cell responses
analyze_cell_responses(cell_responses)
