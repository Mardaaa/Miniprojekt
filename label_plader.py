import cv2
import glob


def load_image(image_file, image_file_number):
    image = cv2.imread(image_file[image_file_number])
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image  

image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
image_file_number = 0

while True:
    key = cv2.waitKey(0)
    if key == 100:
        image_file_number += 1
        print("pressed")
    if key == 27:
        break

print(image_file_number)


# while image_file_number < len(image_files):
#     image = load_image(image_files, image_file_number)
#     key = cv2.waitKey(0)
#     if key == 27: # ESC
#         break
#     elif key == 100: # 'd' key
#         image_file_number +=1
#         cv2.destroyAllWindows()  # Close previous image window
#     elif key == 32: # Space key
#         image_file_number = (image_file_number - 1) % len(image_files)
#         cv2.destroyAllWindows()  # Close previous image window

# cv2.destroyAllWindows()
