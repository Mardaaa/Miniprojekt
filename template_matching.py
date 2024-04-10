import glob
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

def TemplateMatching(image_files, template_files):
    for image_file in image_files:
        image = cv2.imread(image_file, cv2.COLOR_BGR2RGB)

        # Loop over each template
        for template_file in template_files:
            template = cv2.imread(template_file, cv2.COLOR_BGR2RGB)
            # Save the template dimensions
            W, H = template.shape[:2]  
            
            # Perform template matching
            match = cv2.matchTemplate(image=image, templ=template, method=cv2.TM_CCOEFF_NORMED)
            
            # Define a minimum threshold
            thresh = 0.7
            
            # Select rectangles with confidence greater than threshold
            (y_points, x_points) = np.where(match >= thresh)
            
            # Initialize our list of bounding boxes
            boxes = []
            
            # Store coordinates of each bounding box
            for (x, y) in zip(x_points, y_points):
                # Update our list of boxes
                boxes.append((x, y, x + W, y + H))
            
            # Apply non-maxima suppression to the rectangles
            boxes = non_max_suppression(np.array(boxes))
            
            # Loop over the final bounding boxes
            for (x1, y1, x2, y2) in boxes:
                # Draw the bounding box on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Show the final output
        cv2.imshow("After NMS", image)
        cv2.waitKey(0)
        
        # Destroy all the windows
        cv2.destroyAllWindows()

def main():
    image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
    template_files = glob.glob("Crown images/*.jpg")

    TemplateMatching(image_files, template_files)

if __name__ == '__main__':
    main()
