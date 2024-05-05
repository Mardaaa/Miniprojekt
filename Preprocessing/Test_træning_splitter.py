import glob
import random
import os
import shutil
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import csv
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
    # average_hsv = hsv_image.median(axis=0).median(axis=0)
        # Calculate the median HSV values
    median_hsv = np.median(hsv_image, axis=(0,1))
    return median_hsv



def convert_hsv_to_csv(list_subfolders):
    col_names = ['H_median', 'S_median', 'V_median', 'label', 'name_pic']
    df = pd.DataFrame(columns=col_names)
    
    for subfolder in list_subfolders:
        image_folder = glob.glob(f"Categories/{subfolder}/*png")
        
        for image_file in image_folder:
            if image_file.endswith(('.jpg', '.png', '.jpeg')):
                h_mean, s_mean, v_mean = calculate_average_hsv(image_file)
                name_pic = os.path.basename(image_file)
                df = df._append({'H_median': h_mean, 'S_median': s_mean, 'V_median': v_mean, 'label': subfolder, 'name_pic': name_pic}, ignore_index=True)
    
        # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def list_subfolders(source_folder):
    """
    Description:
    ---------------------------
    A function that finds the name of subfolders within a source folder.
    Subfolders are the different cranes, each containing CSV-files.
    ---------------------------
    

    Parameters:
    ---------------------------
    - source_folder: Path to the source folder
    ---------------------------

    Returns:
    ---------------------------
    List of subfolder names
    ---------------------------
    """

    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print(f"The source folder '{source_folder}' does not exist.")
        return

    # Get a list of all items (files and folders) in the source folder
    items = os.listdir(source_folder)

    # Filter out only the subfolders
    subfolders = [item for item in items if os.path.isdir(os.path.join(source_folder, item))]

    # Print the names of the subfolders
    if subfolders:
        print("Subfolders within", source_folder, ":")
        for folder in subfolders:
            print(folder)
    else:
        print("There are no subfolders within", source_folder)
    return subfolders


image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg") # For splitting the data
source_folder = "Categories/" # For calculating HSV-values for tiles

if __name__ == '__main__':
    # data_splitter(image_files)
    list_of_subfolders = list_subfolders(source_folder)
    list_of_subfolders = list_of_subfolders[1:]
    df = convert_hsv_to_csv(list_of_subfolders)
    print(df)
        # Save DataFrame to CSV file
    df.to_csv('hsv_values.csv', index=False)


