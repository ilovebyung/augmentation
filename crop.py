'''
 	converted_keras_AB 타입 구분: 화면에서 2/4~3/4 크기의 ROI 추출 
	0 A
	1 B
'''

import glob
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

source = 'C:\\data\\AB\\A'
target = 'C:\\data\\AB\\A_resized'

source = 'C:\\data\\AB\\B'
target = 'C:\\data\\AB\\B_resized'

## save ROI
os.chdir(source)
for file in os.listdir(source):
    image = plt.imread(file)
    height, width, channel = image.shape
 
    # # cropping 2
    # width_start = np.int(width*2/4)
    # width_end = np.int(width*3/4)
    # roi = image[:, width_start:width_end,:]
    # roi = Image.fromarray(roi)
    # roi.save(f'roi_2_{file}') 

    # cropping 3
    width_start = np.int(width*3/4)
    width_end = np.int(width*4/4)
    roi = image[:, width_start:width_end,:]
    roi = Image.fromarray(roi)
    roi.save(f'roi_2_{file}')

## save resized file    
os.chdir(source) 
dim = (224,224)
# Resize files
for file in glob.glob('*.jpg'):
    image = cv2.imread(file)
    file_name = f'{target}\\{file}'
    print(file_name)
    # resize image
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(file_name, resized_image)
