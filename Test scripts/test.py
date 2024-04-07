import numpy as np

def count_points(field):
    # Create a set to store visited tiles
    visited = set()

    # Initialize points count
    total_points = 0

    # Define directions (up, down, left, right)
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Iterate through each tile
    for i in range(len(field)):
        for j in range(len(field[0])):
            # If the tile is unvisited and not an empty tile
            if (i, j) not in visited and field[i][j] != 0:
                # Perform depth-first search to find connected tiles
                stack = [(i, j)]
                region_points = 0
                while stack:
                    x, y = stack.pop()
                    # If the tile is unvisited and not an empty tile
                    if (x, y) not in visited and field[x][y] != 0:
                        visited.add((x, y))
                        region_points += field[x][y]
                        # Check neighboring tiles
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < len(field) and 0 <= ny < len(field[0]) and (nx, ny) not in visited:
                                stack.append((nx, ny))
                total_points += region_points * count_crowns(region_points)

    return total_points

def count_crowns(region_points):
    # Calculate points based on the number of crowns
    return region_points

# Example playing field (5x5 grid)
playing_field = [
    [0, 0, 0, 0, 0],
    [0, 3, 4, 5, 0],
    [0, 0, 0, 6, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

# Count points
points = count_points(playing_field)
print("Total points:", points)
