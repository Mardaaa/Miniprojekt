import os
import pandas as pd
import glob
import numpy as np
import cv2
from collections import defaultdict
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from imutils.object_detection import non_max_suppression

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
RFC = RF(n_estimators=10, random_state=42)
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

def count_crowns_in_blobs(crown_coordinates, components, results_list, image_name):
    for tile_type, blobs in components.items():
        for blob_index, blob in enumerate(blobs):
            count = sum(1 for crown in crown_coordinates if crown in blob)
            results_list.append({
                "Image": image_name,
                "Tile Type": tile_type,
                "Blob": blob_index + 1,
                "Tiles": len(blob),
                "Crowns": count,
                "Score": count * len(blob)  # Optionally calculate score here if needed elsewhere
            })

if __name__ == '__main__':
    all_results = []
    for board_image_path in board_image_files:
        board_img = cv2.imread(board_image_path)
        image_name = os.path.basename(board_image_path)  # Extract filename for identification
        tiles = split_into_tiles(board_img)
        classified_tile_types = classify_tiles(tiles, RFC)
        components = find_connected_components(classified_tile_types)
        crown_coordinates = TemplateMatching([board_image_path], template_files)
        count_crowns_in_blobs(crown_coordinates, components, all_results, image_name)

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('/Users/jens-jakobskotingerslev/Documents/GitHub/Miniprojekt/Scripts/Resultater2.csv', index=False)
