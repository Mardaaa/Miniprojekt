image_name = "2.jpg"

lst = ["2.jpg_1_0.png"]

# Check if any element in lst contains the substring "2.jpg"
contains_image_name = any(image_name in element for element in lst)
print(contains_image_name)  # This will print True