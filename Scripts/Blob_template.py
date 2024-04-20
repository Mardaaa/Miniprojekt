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
from imutils.object_detection import non_max_suppression


# Paths to CSV and image directories
csv_folder_path = '/Users/jens-jakobskotingerslev/Desktop/Miniprojekt semester 2/HSV values/All'
tiles_folder_path = '/Users/jens-jakobskotingerslev/Desktop/Miniprojekt semester 2/Categories/Alle'
board_image_path = '/Users/jens-jakobskotingerslev/Desktop/Miniprojekt semester 2/King Domino dataset/4.jpg'
template_files_path = '/Users/jens-jakobskotingerslev/Desktop/Miniprojekt semester 2/Scripts/Crown images/*.jpg'
image_files_path = '/Users/jens-jakobskotingerslev/Desktop/Miniprojekt semester 2/Billeder/*.jpg'

# Glob patterns to load files
csv_files = glob.glob(os.path.join(csv_folder_path, '*.csv'))
tile_image_files = glob.glob(os.path.join(tiles_folder_path, '*.png'))
board_img = cv2.imread(board_image_path)
template_files = glob.glob(template_files_path)
image_files = glob.glob(image_files_path)
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
RFC = RF(n_estimators=25, random_state=42)
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


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def TemplateMatching(image_files, template_files):
    tile_size = 100  # Size of each tile

    for image_file in image_files:
        image = cv2.imread(image_file)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert BGR image to HSV
        num_blobs = 0  # Counter for the number of blobs
        blob_centers = []  # List to store the centers of detected blobs
    
        
        # Loop over each template
        for template_file in template_files:
            template = cv2.imread(template_file)
            template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            # Save the template dimensions
            W, H = template.shape[:2]  
            
            # Perform template matching
            match = cv2.matchTemplate(image=image_hsv, templ=template_hsv, method=cv2.TM_CCOEFF_NORMED)
            
            # Define a minimum threshold
            thresh = 0.62
            
            # Select rectangles with confidence greater than threshold
            (y_points, x_points) = np.where(match >= thresh)
            
            # Initialize our list of bounding boxes
            boxes = []
            
            # Store coordinates of each bounding box
            for (x, y) in zip(x_points, y_points):
                # Update our list of boxes
                boxes.append((x, y, x + W, y + H))
                num_blobs += 1  # Increment the counter for each blob
                # Calculate the center of the detected blob
                center = ((x + x + W) // 2, (y + y + H) // 2)
                # Check if the center is close to the centers of previously detected blobs
                close_to_existing = False
                for blob_center in blob_centers:
                    if distance(center, blob_center) < 20:  # Adjust the threshold distance as needed
                        close_to_existing = True
                        break
                # If not close to existing blob centers, add it to the list
                if not close_to_existing:
                    blob_centers.append(center)
            
            # Apply non-maxima suppression to the rectangles
            boxes = non_max_suppression(np.array(boxes))
            
            # Loop over the final bounding boxes
            for (x1, y1, x2, y2) in boxes:
                # Draw the bounding box on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        crown_cordinates = []
        for (x1, y1) in blob_centers:
                            # Calculate tile coordinates
            tile_x = y1 // tile_size
            tile_y = x1 // tile_size
            crown_cordinates.append((tile_x, tile_y))
        print(f"Crown cordinates {crown_cordinates} and number of crowns {len(blob_centers)}")
    return crown_cordinates

def count_crowns_in_blobs(crown_coordinates, components):
    crown_count_per_blob = {}

    for tile_type, blobs in components.items():
        for blob_index, blob in enumerate(blobs):
            count = 0

            for crown in crown_coordinates:
                if crown in blob:
                    count += 1
            crown_count_per_blob[(tile_type, blob_index)] = count
            print(f"Tile Type: {tile_type}, Blob {blob_index + 1}, Crowns: {count}")

    return crown_count_per_blob

def calculate_total_score(crown_coordinates, components):
    total_score = 0
    component_scores = {}

    for tile_type, blobs in components.items():
        for blob_index, blob in enumerate(blobs):
            crown_count = 0

            for crown in crown_coordinates:
                if crown in blob:
                    crown_count += 1

            blob_score = crown_count * len(blob)
            component_scores[(tile_type, blob_index)] = blob_score
            total_score += blob_score
            print(f"Tile Type: {tile_type}, Blob {blob_index + 1}, Tiles: {len(blob)}, Crowns: {crown_count}, Score: {blob_score}")

    print(f"Total Score: {total_score}")
    return total_score, component_scores

if __name__ == '__main__':
    tiles = split_into_tiles(board_img)
    classified_tile_types = classify_tiles(tiles, RFC)
    components = find_connected_components(classified_tile_types)
    crown_cordinates = TemplateMatching(image_files, template_files)    
    crown_count_per_blob = count_crowns_in_blobs(crown_cordinates, components)
    calculate_total_score(crown_cordinates, components)
    report = classification_report(y_test, y_pred)
    #print(report)
