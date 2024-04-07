import cv2
import glob
import matplotlib.pyplot as plt

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

# Load the image
image_files = glob.glob("King Domino dataset/Cropped and perspective corrected boards/*.jpg")
image = cv2.imread(image_files[0])

fields = divide_into_5x5_fields(image)
# print(fields)

# Visualize each field
for i, field in enumerate(fields):
    plt.imshow(cv2.cvtColor(field, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for Matplotlib
    plt.title(f'Field {i+1}')
    plt.axis('off')
    plt.show()