import pandas as pd
import os
import glob
import numpy as np
import cv2
from collections import defaultdict
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# File paths
board_images_folder = ("Preprocessing/test data/*.jpg")
csv_file = "CSV_filer/hsv_values.csv"

# Glob patterns to load files
board_image_files = glob.glob(os.path.join(board_images_folder))
df = pd.read_csv(csv_file)

# List of test images
test_list = ["1.jpg", "6.jpg", "9.jpg", "13.jpg", "14.jpg", "18.jpg", "19.jpg", "20.jpg",
              "26.jpg", "35.jpg", "38.jpg", "40.jpg", "50.jpg", "67.jpg", "69.jpg"]


# Make X and y data
df = df[~df['name_pic'].isin(test_list)]
df.drop(['name_pic'], axis=1, inplace=True)
X = df.drop(['label'], axis=1)
y = df['label']

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
RFC = RF(criterion='entropy',n_estimators=10, random_state=42)
RFC.fit(X_train, y_train)
y_pred = RFC.predict(X_test)

# compute median hsv
def compute_median_hsv(image_tile):
    hsv_tile = cv2.cvtColor(image_tile, cv2.COLOR_BGR2HSV)
    return np.median(hsv_tile, axis=(0, 1))

# classify tile
def classify_tile(average_hsv, classifier):
    tile_type = classifier.predict([average_hsv])
    return tile_type[0]

# classify tiles
def classify_tiles(tiles, classifier):
    classified_tiles = []
    # for loop to classify tiles
    for tile in tiles:
        average_hsv = compute_median_hsv(tile)
        tile_type = classify_tile(average_hsv, classifier)
        classified_tiles.append(tile_type)
    return classified_tiles

# Split image into tiles
def split_into_tiles(board_image, grid_size=(5, 5)):
    tiles = []
    # Calculate tile size
    tile_height = board_image.shape[0] // grid_size[0]
    tile_width = board_image.shape[1] // grid_size[1]
    # for loop to split image into tiles
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            start_row = row * tile_height
            start_col = col * tile_width
            tile = board_image[start_row:start_row + tile_height, start_col:start_col + tile_width]
            tiles.append(tile)
    return tiles

# Find connected components
def find_connected_components(classified_tiles, grid_size=(5, 5)):
    # Initialize empty set and dictionary
    visited = set()
    components = defaultdict(list)

    # Define neighbors
    def dfs(row, col, tile_type):
        # Initialize stack and component
        stack = [(row, col)]
        component = set()
        # while loop to find connected components
        while stack:
            # pop from queue
            r, c = stack.pop()
            # if statement to check if visited
            if (r, c) in visited:
                continue
            # Add to visited and component
            visited.add((r, c))
            component.add((r, c))
            # for loop to find neighbors
            for nr, nc in neighbors(r, c):
                # if statement to check if visited and classified tiles
                if (nr, nc) not in visited and classified_tiles[nr * grid_size[1] + nc] == tile_type:
                    stack.append((nr, nc))
        return component

    def neighbors(r, c):
        # for loop to find neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Calculate new row and column
            nr, nc = r + dr, c + dc
            # if statement to check if new row and column is within grid
            if 0 <= nr < grid_size[0] and 0 <= nc < grid_size[1]:
                # yield new row and column
                yield nr, nc

    # for loop to find connected components
    for row, col in itertools.product(range(grid_size[0]), range(grid_size[1])):
        # if statement to check if visited
        if (row, col) not in visited:
            # find current type
            current_type = classified_tiles[row * grid_size[1] + col]
            # find connected components
            component = dfs(row, col, current_type)
            # append to components
            components[current_type].append(component)
    return components

# Main function
if __name__ == '__main__':
    # Initialize empty list
    all_results = []
    # for loop to iterate over board images
    for board_image_path in board_image_files:
        board_img = cv2.imread(board_image_path)
        image_name = os.path.basename(board_image_path)
        tiles = split_into_tiles(board_img)
        classified_tile_types = classify_tiles(tiles, RFC)
        components = find_connected_components(classified_tile_types)
        print(components)

