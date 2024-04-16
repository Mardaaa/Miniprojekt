import glob
import random
import os
import shutil
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import csv

def data_splitter(image_files, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)

    # Move training data to "training data" folder
    for file in train_files:
        filename = os.path.basename(file)
        dest_path = os.path.join("training data", filename)
        os.makedirs("training data", exist_ok=True)
        shutil.copy(file, dest_path)
    
    for file in test_files:
        filename = os.path.basename(file)
        dest_path = os.path.join("test data", filename)
        os.makedirs("test data", exist_ok=True)
        shutil.copy(file, dest_path)

def calculate_average_hsv(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate the average HSV values
    average_hsv = hsv_image.mean(axis=0).mean(axis=0)
    return average_hsv



def convert_hsv_to_csv(image_folder):
    col_name = "Field"
    df = pd.DataFrame()
    for image in image_folder:
        calculate_average_hsv(image)

        


image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
image_folder = glob.glob("Categories/Field/*.png")


if __name__ == '__main__':
    # data_splitter(image_files)
    print(image_folder)
    # var = calculate_average_hsv(image_folder[0])
    print(var)