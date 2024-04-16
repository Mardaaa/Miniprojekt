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



def convert_hsv_to_csv(image_folder, label):
    col_names = ['H_mean', 'S_mean', 'V_mean', 'label', 'name_pic']
    df = pd.DataFrame(columns=col_names)
    
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            h_mean, s_mean, v_mean = calculate_average_hsv(image_path)
            name_pic = image_file.split('\\')[-1]
            df = df.append({'H_mean': h_mean, 'S_mean': s_mean, 'V_mean': v_mean, 'label': label, 'name_pic': name_pic}, ignore_index=True)
    
    return df
        


image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
field_folder = glob.glob("Categories/Field/*.png")
field_folder = glob.glob("Categories/Field/*.png")
field_folder = glob.glob("Categories/Field/*.png")
field_folder = glob.glob("Categories/Field/*.png")
field_folder = glob.glob("Categories/Field/*.png")
field_folder = glob.glob("Categories/Field/*.png")
field_folder = glob.glob("Categories/Field/*.png")
field_folder = glob.glob("Categories/Field/*.png")


if __name__ == '__main__':
    # data_splitter(image_files)
    # print(image_folder)
    # var = calculate_average_hsv(image_folder[0])
    var = convert_hsv_to_csv(image_folder,'Field')
    print(var)