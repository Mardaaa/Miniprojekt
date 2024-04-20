import pandas as pd
import os
import glob
import numpy as np
import cv2
from collections import defaultdict
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF


# Paths to CSV and image directories
board_images_folder = '/Users/jens-jakobskotingerslev/Desktop/Ren mini projekt/test data'
csv_file = '/Users/jens-jakobskotingerslev/Desktop/Ren mini projekt/hsv_values.csv'
template_files_path = '/Users/jens-jakobskotingerslev/Desktop/Miniprojekt semester 2/Scripts/Crown images'

# Glob patterns to load files
board_image_files = glob.glob(os.path.join(board_images_folder, '*.jpg'))
template_files = glob.glob(os.path.join(template_files_path, '*.jpg'))
df = pd.read_csv(csv_file)


# Model
df.drop(['name_pic'], axis=1, inplace=True)
X = df.drop(['label'], axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
RFC = RF(criterion='entropy',n_estimators=10, random_state=42)
RFC.fit(X_train, y_train)
y_pred = RFC.predict(X_test)

def compute_median_hsv(image_tile):
    hsv_tile = cv2.cvtColor(image_tile, cv2.COLOR_BGR2HSV)
    return np.median(hsv_tile, axis=(0, 1))

def classify_tile(average_hsv, classifier):
    tile_type = classifier.predict([average_hsv])
    return tile_type[0]

def classify_tiles(tiles, classifier):
    classified_tiles = []
    for tile in tiles:
        average_hsv = compute_median_hsv(tile)
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
    all_results = []
    for board_image_path in board_image_files:
        board_img = cv2.imread(board_image_path)
        image_name = os.path.basename(board_image_path)
        tiles = split_into_tiles(board_img)
        classified_tile_types = classify_tiles(tiles, RFC)
        components = find_connected_components(classified_tile_types)
