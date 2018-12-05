import glob
import cv2
import csv
import pytesseract as tess

from modifyImage import Modify

data_path = 'GTSRB\Final_Training\SpeedLimitSigns'

csv_paths = glob.glob(data_path + '/*/*.csv')
data = []
im_size = (100,100)
    
im_paths = glob.glob(data_path + '/*/*.ppm')
    
for path in csv_paths:
    
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter = ';')
        next(reader, None)
        csvdata = list(reader)
        for i in range(len(csvdata)):
            data.append(csvdata[i])
      
images = [cv2.imread(im_paths[50])]
windowNames = []

for image in images:
    image = Modify(image)
    