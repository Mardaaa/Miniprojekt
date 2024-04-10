import glob
import cv2

image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
image = cv2.imread(image_files[0])

