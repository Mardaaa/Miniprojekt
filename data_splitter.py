import glob
import random
import os
import shutil
from sklearn.model_selection import train_test_split


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



image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")

if __name__ == '__main__':
    data_splitter(image_files)