from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import cv2

datagen = ImageDataGenerator(
    rotation_range=5,  # Random rotation between 0 and 5
    width_shift_range=0.2,  # % shift
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant', cval=125)  # Also try nearest, constant, reflect, wrap

image = cv2.imread('0.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
shape = image.shape
height = shape[0]
width = shape[1]

# Array with shape (1, height, width, 3)
x = image.reshape((1, ) + image.shape)

i = 0
for batch in datagen.flow(x, batch_size=6,
                          save_to_dir='augmented',
                          save_prefix='aug',
                          save_format='jpg'):
    i += 1
    if i > 10:
        break  # otherwise the generator would loop indefinitely
