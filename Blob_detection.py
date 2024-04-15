import os
import glob
import csv
import numpy as np
import cv2
from collections import defaultdict
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF

# Paths to CSV and image directories
csv_folder_path = '/Users/jens-jakobskotingerslev/Desktop/Miniprojekt semester 2/HSV values/All'
tiles_folder_path = '/Users/jens-jakobskotingerslev/Desktop/Miniprojekt semester 2/Categories/Alle'
board_image_path = '/Users/jens-jakobskotingerslev/Desktop/Miniprojekt semester 2/King Domino dataset/26.jpg'

# Glob patterns to load files
csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))
tile_image_files = glob.glob(os.path.join(tiles_folder_path, '*.png'))
board_img = cv2.imread(board_image_path)

# Load tile HSV values
def load_tile_hsv_values(csv_files):
    tile_hsv_values = []
    tile_types = []
    for csv_file in csv_files:
        with open(csv_file, newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                _, hue, saturation, value = row
                tile_hsv_values.append([float(hue), float(saturation), float(value)])
                tile_type = os.path.basename(csv_file).split('_')[0]
                tile_types.append(tile_type)
    return np.array(tile_hsv_values), tile_types


def compute_average_hsv(image_tile):
    hsv_tile = cv2.cvtColor(image_tile, cv2.COLOR_BGR2HSV)
    return hsv_tile.mean(axis=0).mean(axis=0)

def classify_tile(average_hsv, classifier):
    tile_type = classifier.predict([average_hsv])
    return tile_type[0]

# Model
tile_hsv_values, tile_types = load_tile_hsv_values(csv_files)
X_train, X_test, y_train, y_test = train_test_split(tile_hsv_values, tile_types, test_size=0.2, random_state=42)
RFC = RF(n_estimators=20, random_state=42)
RFC.fit(X_train, y_train)
y_pred = RFC.predict(X_test)

# blob detection
def classify_tiles(tiles, classifier):
    classified_tiles = []
    for tile in tiles:
        average_hsv = compute_average_hsv(tile)
        tile_type = classify_tile(average_hsv, classifier)
        classified_tiles.append(tile_type)
    return classified_tiles

def split_into_tiles(board_image, grid_size=(5, 5)):
    tiles = []
    tile_height = board_image.shape[0] // grid_size[0]
    tile_width = board_image.shape[1] // grid_size[1]
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            start_row = row * tile_height
            start_col = col * tile_width
            tile = board_image[start_row:start_row + tile_height, start_col:start_col + tile_width]
            tiles.append(tile)
    return tiles

def find_connected_components(classified_tiles, grid_size=(5, 5)):
    visited = set()
    components = defaultdict(list)

    def dfs(row, col, tile_type):
        stack = [(row, col)]
        component = set()
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            component.add((r, c))
            for nr, nc in neighbors(r, c):
                if (nr, nc) not in visited and classified_tiles[nr * grid_size[1] + nc] == tile_type:
                    stack.append((nr, nc))
        return component

    def neighbors(r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size[0] and 0 <= nc < grid_size[1]:
                yield nr, nc

    for row, col in itertools.product(range(grid_size[0]), range(grid_size[1])):
        if (row, col) not in visited:
            current_type = classified_tiles[row * grid_size[1] + col]
            component = dfs(row, col, current_type)
            components[current_type].append(component)
    return components

if __name__ == '__main__':

    tiles = split_into_tiles(board_img)
    print("Tiles split:", len(tiles))

    classified_tile_types = classify_tiles(tiles, RFC)
    print("Classification completed. First few types:", classified_tile_types[:])

    components = find_connected_components(classified_tile_types)
    for tile_type, comps in components.items():
        print(f"Tile Type: {tile_type}, Number of Components: {len(comps)}")
        for comp in comps:
            print(f"Component: {comp}")

    # Evaluation
    report = classification_report(y_test, y_pred)
    print(report)
