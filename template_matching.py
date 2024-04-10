import glob
import cv2

image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
image = cv2.imread(image_files[0])

def TemplateMatching(image, templates):
    pass




def main():
    image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
    
    image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)


if __name__ == '__main__':
    main()