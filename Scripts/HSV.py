import cv2
import os
import csv
import re

def calculate_average_hsv(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate the average HSV values
    average_hsv = hsv_image.mean(axis=0).mean(axis=0)
    return average_hsv

def numeric_order(filename):
    # Extract the numeric part of the filename before the first underscore or dot
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def process_images(folder_path):
    hsv_values = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            average_hsv = calculate_average_hsv(file_path)
            hsv_values.append([filename, *average_hsv])

    # Sort the list by the numeric part of the filename in ascending order
    hsv_values.sort(key=lambda x: numeric_order(x[0]))

    with open('Forrest_hsv_values.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Average Hue', 'Average Saturation', 'Average Value'])
        writer.writerows(hsv_values)

    print("HSV values have been successfully written to 'image_hsv_values.csv'.")

# Replace 'path_to_your_folder' with the path to the folder containing your images
folder_path = '/Users/jens-jakobskotingerslev/Desktop/Miniprojekt semester 2/Categories/Swamp'
process_images(folder_path)
