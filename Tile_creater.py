import cv2 as cv
import os

# Paths
original_folder_path = '/Users/jens-jakobskotingerslev/Documents/GitHub/Miniprojekt/King Domino dataset/Cropped and perspective corrected boards'
base_cropped_folder_path = '/Users/jens-jakobskotingerslev/Documents/GitHub/Miniprojekt/Cropped folder'

# laver 74 nye folders
def createFolder(base_directory, number_of_folders):
    for i in range(1, number_of_folders + 1):
        directory = os.path.join(base_directory, f'cropped_{i}')
        if not os.path.exists(directory):
            os.makedirs(directory)

# Laver og sorterer filer i "original_folder_path"
def crop_and_save_images(original_folder_path, base_cropped_folder_path, number_of_images):
    image_files = [f for f in os.listdir(original_folder_path) if os.path.isfile(os.path.join(original_folder_path, f))]
    image_files.sort() 

    # Looper over alle billeder i "original_folder_path"
    for index, image_file in enumerate(image_files[:number_of_images], start=1):
        file_path = os.path.join(original_folder_path, image_file)
        img = cv.imread(file_path)
        img = cv.resize(img, (500, 500)) 

        # Opdeler i tiles
        img_height, img_width = img.shape[:2]
        M, N = img_height // 5, img_width // 5
        counter = 1

        # Looper over tiles of tilføjer dem til mappe 
        for y in range(0, img_height, M):
            for x in range(0, img_width, N):
                tiles = img[y:y+M, x:x+N]
                cropped_folder_path = os.path.join(base_cropped_folder_path, f'cropped_{index}')
                filename = os.path.join(cropped_folder_path, f'square_{counter}.jpg')
                cv.imwrite(filename, tiles)
                counter += 1

# Cropper og gemmer billeder
number_of_folders,number_of_images = 74,74
createFolder(base_cropped_folder_path, number_of_folders)
crop_and_save_images(original_folder_path, base_cropped_folder_path, number_of_images)
