import cv2
import glob
import os

file = 'D:/Data/Root/root.jpg'
image = cv2.imread(file, 0)
height, width = image.shape

dim = (int(width/2), int(height/2))

source = 'D:/Data/Root/augmented'
target = 'D:/Data/Root/augmented'

# Move to source directory
os.chdir(source)

# Resize files
for file in glob.glob('*.jpg'):
    image = cv2.imread(file, 0)
    file_name = f'{target}//{file}'
    print(file_name)
    # resize image
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(file_name, resized_image)
