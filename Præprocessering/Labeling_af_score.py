import cv2
import glob
import keyboard
import os
import csv

# def load_image(image_file, image_file_number):
#     image = cv2.imread(image_file[image_file_number])
#     # cv2.imshow(f"Image: {image_file_number+1}", image)
#     return image  

image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
image_file_number = 0

scores = [] # List to store scores for each image
flag = True

while image_file_number < len(image_files) and flag:
    # image = load_image(image_files, image_file_number)
    # while True:
    if keyboard.is_pressed('esc'):
        break
    
    # current_state_d = keyboard.is_pressed('d')
    if image_file_number < len(image_files):
        image_scores = []  # List to store scores for the current image
        # print(f"Enter scores for Image {image_file_number+1}:")
        # Collect scores for each attribute
        score_lake = input(f"Lake score for image{image_file_number+1}.jpg: ")
        score_forest = input(f"Forest score for image{image_file_number+1}.jpg: ")
        score_grassland = input(f"Grassland score for image{image_file_number+1}.jpg: ")
        score_field = input(f"Field score for image{image_file_number+1}.jpg: ")
        score_swamp = input(f"Swamp score for image{image_file_number+1}.jpg: ")
        score_mine = input(f"Mine score for image{image_file_number+1}.jpg: ")
        # Append the scores to the image_scores list
        image_scores.extend([int(score_lake), int(score_forest), int(score_grassland), int(score_field), int(score_swamp), int(score_mine)])
        # Append the image number and its corresponding scores to the scores list
        sum_scores = sum(image_scores)
        scores.append([image_file_number+1] + image_scores + [sum_scores])
        image_file_number += 1
        flag = input("Cotinue? y/n: ")
        if flag == 'n':
            flag = False
        if flag == 'y':
            flag = True



def save_file(output_path, lst_name, file_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, file_name + ".csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Image Number", "Lake Score", "Forest Score", "Grassland Score", "Field Score", "Swamp Score", "Mine Score", "Total score"])
        # Write data
        for item in lst_name:
            writer.writerow(item)
    print("Saved labeled dataset")

output_path = "Labeled plader"
save_file(output_path, scores, "Labels")