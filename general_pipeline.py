import cv2
import numpy as np
import glob

"""
Der antages at der benyttes ML-classification til at classificere de forskellige terræntyper
1. Opdel spillebrættet op i 5x5
2. Ekstraher den gennemsnitlige HSV-værdi ud fra angivne tile.
3. Brug ML-modellen til at predicte, hvilken terræntype det er.
4. Se om terræntyperne hænger sammen
5. Identificer kroner vha. template matching
6. Tæl point
"""

# Read the image
# image = cv2.imread('king_domino_field.jpg')

def divide_into_5x5_fields(image):
    # Divide the image into 5x5 fields
    fields = []
    height, width = image.shape[:2]
    field_height = height // 5
    field_width = width // 5
    for i in range(5):
        for j in range(5):
            field = image[i*field_height:(i+1)*field_height, j*field_width:(j+1)*field_width]
            fields.append(field)
    return fields # Length of fields is 25


# Define terrain types and their corresponding HSV thresholds
terrain_thresholds = {
    "Forest": {"h_min": 30, "h_max": 90, "s_min": 40, "s_max": 255, "v_min": 20, "v_max": 255},
    "Lake": {"h_min": 90, "h_max": 130, "s_min": 40, "s_max": 255, "v_min": 20, "v_max": 255},
    "Wheat Field": {"h_min": 20, "h_max": 40, "s_min": 40, "s_max": 255, "v_min": 20, "v_max": 255},
    # Add more terrain types and their thresholds here
}

def get_terrain(hsv_value):
    h, s, v = hsv_value
    for terrain, thresholds in terrain_thresholds.items():
        h_min, h_max = thresholds["h_min"], thresholds["h_max"]
        s_min, s_max = thresholds["s_min"], thresholds["s_max"]
        v_min, v_max = thresholds["v_min"], thresholds["v_max"]
        if h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max:
            return terrain
    return "Unknown"  # If no match is found

def terrains_in_image(image):
    fields = divide_into_5x5_fields(image)
    terrains_2d = []
    for i in range(5):
        row_terrains = []
        for j in range(5):
            field = fields[i * 5 + j]
            hsv_field = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
            mean_hsv = cv2.mean(hsv_field)[:3]  # Calculate mean HSV values
            terrain = get_terrain(mean_hsv)
            row_terrains.append(terrain)
        terrains_2d.append(row_terrains)
    return terrains_2d

# Load the image
image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
image = cv2.imread(image_files[0])

terrains = terrains_in_image(image)
print(terrains)
# # Tile Detection (e.g., contour detection or template matching)
# # Use your preferred method to detect individual tiles in the image
# tiles = get_terrain(hsv_values)


# def identify_regions(tiles):
#     # Måske contours (edges) for at finde BLOBS idk
#     pass


# # Region Identification (e.g., using connected component analysis or clustering)
# regions = identify_regions(tiles)

# total_points = 0

# def count_crowns(region):
#     pass

# # Iterate over each region
# for region in regions:
#     # Count crowns in the region
#     crowns = count_crowns(region)

#     # Count tiles in the region
#     tiles_count = len(region)

#     # Calculate points for the region
#     region_points = tiles_count * crowns

#     # Accumulate total points
#     total_points += region_points

# print("Total points:", total_points)
