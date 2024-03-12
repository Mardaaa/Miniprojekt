import cv2
import glob
import keyboard
import os
import csv

def load_image(image_file, image_file_number):
    image = cv2.imread(image_file[image_file_number])
    cv2.imshow(f"Image: {image_file_number+1}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image  

image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
image_file_number = 0

scores = [] # 2D list: Image number and score

while image_file_number < len(image_files):
    image = load_image(image_files, image_file_number)


    if keyboard.is_pressed('esc'):
        break

    current_state_d = keyboard.is_pressed('d')
    if current_state_d and image_file_number < len(image_files):
        score_input = input(f"Type in score for image.{image_file_number+1}.jpg: ")
        try:
            score = int(score_input)
            scores.append([image_file_number+1, score])
            image_file_number += 1
        except ValueError:
            print("Please enter a valid integer score.")

    current_state_a = keyboard.is_pressed('a')
    if current_state_a and image_file_number > 0:
        image_file_number -= 1

cv2.destroyAllWindows()

def save_file(output_path, lst_name, file_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, file_name + ".csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Number", "Score"])
        for item in lst_name:
            writer.writerow(item)
    print("Saved labeled dataset")

output_path = "Labeled plader"
save_file(output_path, scores, "Labels")
