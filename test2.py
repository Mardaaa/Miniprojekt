import cv2
import glob
import matplotlib.pyplot as plt
import csv

# Function to read CSV file containing terrain type labels and HSV values
def read_csv_file(csv_file):
    hsv_values = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            category = row[0]
            hsv_values.setdefault(category, []).append([float(val) for val in row[1:]])
    return hsv_values


def divide_into_5x5_fields(image):
    # Divide the image into 5x5 fields
    fields = []
    height, width = image.shape[:2]
    field_height = height // 5
    field_width = width // 5
    for i in range(5):
        for j in range(5):
            field = image[i*field_height:(i+1)*field_height, j*field_width:(j+1)*field_width]
            fields.append(field)
    return fields

# Function to visualize the HSV color distribution of a field
def visualize_hsv_distribution(field, terrain_type, hsv_values):
    hsv_field = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)  # Convert field to HSV color space
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(field, cv2.COLOR_BGR2RGB))  # Display the original field
    plt.title(f'Field: {terrain_type}')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    h, s, v = cv2.split(hsv_field)
    plt.hist(h.ravel(), bins=180, color='red', alpha=0.5, label='Hue')
    plt.hist(s.ravel(), bins=256, color='green', alpha=0.5, label='Saturation')
    plt.hist(v.ravel(), bins=256, color='blue', alpha=0.5, label='Value')
    plt.title('HSV Color Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Read CSV file containing terrain type labels and HSV values
csv_file = r"M:\UNI Daki\2. semester\Miniprojekt github\Miniprojekt\CSV\combined.csv"  # Replace with the path to your CSV file
hsv_values = read_csv_file(csv_file)

# Load the image
image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
image = cv2.imread(image_files[0])

# Divide the image into 5x5 fields
fields = divide_into_5x5_fields(image)

# Visualize the HSV color distribution for each field
for i, field in enumerate(fields):
    terrain_type = "field"  # Example terrain type, you need to determine the terrain type for each field
    visualize_hsv_distribution(field, terrain_type, hsv_values)