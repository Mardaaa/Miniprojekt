import glob
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def TemplateMatching(image_files, template_files):
    image_counter = 0
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

        for (x1, y1) in blob_centers:
                            # Calculate tile coordinates
            tile_x = x1 // tile_size
            tile_y = y1 // tile_size
            print(f"Crown is on tile ({tile_x}, {tile_y})")
        
        print(f"Number of crowns: {len(blob_centers)}")

        # Show the final output
        cv2.imshow(f"Image {image_counter}", image)
        cv2.waitKey(0)
        
        # Destroy all the windows
        cv2.destroyAllWindows()
        image_counter += 1

def main():
    image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
    template_files = glob.glob("Crown images/*.jpg")

    TemplateMatching(image_files, template_files)

if __name__ == '__main__':
    main()
