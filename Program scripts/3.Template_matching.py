import glob
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

# File paths
image_files_path = 'King Domino dataset/Cropped and perspective corrected boards/*.jpg'
template_files_path = 'Preprocessing/Crown images/*.jpg'

image_files = glob.glob(image_files_path)
template_files = glob.glob(template_files_path)


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def TemplateMatching(image_files, template_files):
    tile_size = 100  # Size of each tile

    for image_file in image_files:
        image = cv2.imread(image_file)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert BGR image to HSV
        num_crowns = 0  # Counter for the number of crowns
        crown_centers = []  # List to store the centers of detected crowns
    
        
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
                num_crowns += 1  # Increment the counter for each crown
                # Calculate the center of the detected crown
                center = ((x + x + W) // 2, (y + y + H) // 2)
                # Check if the center is close to the centers of previously detected crowns
                close_to_existing = False
                for crown_center in crown_centers:
                    if distance(center, crown_center) < 20:  # Adjust the threshold distance as needed
                        close_to_existing = True
                        break
                # If not close to existing crown centers, add it to the list
                if not close_to_existing:
                    crown_centers.append(center)
            
            # Apply non-maxima suppression to the rectangles
            boxes = non_max_suppression(np.array(boxes))
            
            # Loop over the final bounding boxes
            for (x1, y1, x2, y2) in boxes:
                # Draw the bounding box on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        crown_cordinates = []
        for (x1, y1) in crown_centers:
                            # Calculate tile coordinates
            tile_x = y1 // tile_size
            tile_y = x1 // tile_size
            crown_cordinates.append((tile_x, tile_y))
        print(f"Crown cordinates {crown_cordinates} and number of crowns {len(crown_centers)}")


        # Show the final output
        #cv2.imshow("Image with identified crowns", image)
        #cv2.waitKey(0)
        
        # Destroy all the windows
        #cv2.destroyAllWindows()
        
    return crown_cordinates

if __name__ == '__main__':
    TemplateMatching(image_files, template_files)