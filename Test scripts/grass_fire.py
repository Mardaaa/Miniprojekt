# import cv2
# import numpy as np
# import glob

# def grassfire(image):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Threshold the image to get binary image
#     _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    
#     # Initialize labels matrix
#     labels = np.zeros_like(binary_image, dtype=np.uint8)
#     current_label = 1
    
#     # Define 8-connected neighbors
#     neighbors = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if (i != 0 or j != 0)]
    
#     # Perform grass-fire algorithm
#     for y in range(binary_image.shape[0]):
#         for x in range(binary_image.shape[1]):
#             if binary_image[y, x] == 255 and labels[y, x] == 0:
#                 stack = [(x, y)]
#                 while stack:
#                     current_pixel = stack.pop()
#                     labels[current_pixel[1], current_pixel[0]] = current_label
#                     for dx, dy in neighbors:
#                         nx, ny = current_pixel[0] + dx, current_pixel[1] + dy
#                         if 0 <= nx < binary_image.shape[1] and 0 <= ny < binary_image.shape[0] and binary_image[ny, nx] == 255 and labels[ny, nx] == 0:
#                             stack.append((nx, ny))
#                 current_label += 1
    
#     return labels

# # Load the image
# image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
# image = cv2.imread(image_files[0])

# # Apply grass-fire algorithm
# labels = grassfire(image)

# # Convert labels to a color image
# colored_labels = cv2.applyColorMap((labels * 10).astype(np.uint8), cv2.COLORMAP_JET)

# # Display the result
# cv2.imshow('Labels', colored_labels)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
import glob

# Load the image
image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
image = cv2.imread(image_files[0])
# image = cv2.imread('king_domino_field.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to obtain a binary image
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Perform morphological operations to enhance the blobs
kernel = np.ones((25,25),np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Draw bounding boxes around connected components (blobs)
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area > 100:  # Filter small components
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
cv2.imshow("Connected Components", image)
cv2.waitKey(0)
cv2.destroyAllWindows()