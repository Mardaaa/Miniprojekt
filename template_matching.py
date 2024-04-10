import glob
import cv2
from imutils.object_detection

def TemplateMatching(image_file, template):
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
    # save the image dimensions 
    W, H = template.shape[:2] 

    match = cv2.matchTemplate(image=image, templ=template, method=cv2.TM_CCOEFF_NORMED)

    # Define a minimum threshold 
    thresh = 0.4
    # Select rectangles with 
    # confidence greater than threshold 
    (y_points, x_points) = np.where(match >= thresh) 
    
    # initialize our list of bounding boxes 
    boxes = list() 
    
    # store co-ordinates of each bounding box 
    # we'll create a new list by looping 
    # through each pair of points 
    for (x, y) in zip(x_points, y_points): 
        
        # update our list of boxes 
        boxes.append((x, y, x + W, y + H)) 

def main():
    image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
    template_files = glob.glob("Crown images/*.jpg")


    

    cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()