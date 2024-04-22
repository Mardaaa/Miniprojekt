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

# Make X and y data
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

# Count crowns in blobs
def count_crowns_in_blobs(crown_coordinates, components, results_list, image_name):
    # Loop through each tile type and its corresponding list of blobs
    for tile_type, blobs in components.items():
        # Iterate over each blob and its index within the blobs list
        for blob_index, blob in enumerate(blobs):
            # Count how many crown coordinates are present within this specific blob
            count = sum(1 for crown in crown_coordinates if crown in blob)
            # Append a dictionary to results_list containing details about the image,
            # tile type, specific blob, number of tiles in the blob, number of crowns,and a score
            results_list.append({
                "Image": image_name,       
                "Tile Type": tile_type,   
                "Blob": blob_index + 1,   
                "Tiles": len(blob),      
                "Crowns": count,          
                "Score": count * len(blob)
            })

# Main function
if __name__ == '__main__':
    # Initialize list to store results
    all_results = []
    # Loop through each board image
    for board_image_path in board_image_files:
        # Read the board image
        board_img = cv2.imread(board_image_path)
        # Extract the image name
        image_name = os.path.basename(board_image_path) 
        # Split the board image into tiles
        tiles = split_into_tiles(board_img)
        # Classify the tiles using the trained Random Forest Classifier
        classified_tile_types = classify_tiles(tiles, RFC)
        # Find connected components in the classified tiles
        components = find_connected_components(classified_tile_types)
        # Perform template matching to detect crowns and their coordinates
        crown_coordinates = TemplateMatching([board_image_path], template_files)
        # Count the number of crowns in each blob and store the results
        count_crowns_in_blobs(crown_coordinates, components, all_results, image_name)
        
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('/Users/jens-jakobskotingerslev/Documents/GitHub/Miniprojekt/Scripts/Resultater2.csv', index=False)
