import glob
import cv2
import numpy as np






from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Loading train images
images_path = "CamVid/train/"
images = glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpg")
images.sort()

X = []
width = 200
height = 100
for img in images:
    image = cv2.imread(img)
    image = cv2.resize(image, (width, height))
    image = image / np.max(image)
    image = image.astype(np.float32)
    X.append(image)