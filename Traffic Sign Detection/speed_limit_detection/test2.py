import numpy as np
from easydict import EasyDict
import glob
import cv2
import csv
import pytesseract as tess
from docutils.nodes import Labeled
from cv2 import imshow, imread
import tools

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
      
images = []
windowNames = []

test_im = cv2.imread(im_paths[29])
test_im = cv2.resize(test_im, im_size, interpolation = cv2.INTER_CUBIC)
images.append(test_im)
windowNames.append("No change")

normalized_im = np.zeros(im_size)
normalized_im = cv2.normalize(test_im,normalized_im,0, 255, cv2.NORM_MINMAX)
images.append(normalized_im)
windowNames.append("Normalized1")

gray_im = cv2.cvtColor(normalized_im, cv2.COLOR_BGR2GRAY)
images.append(gray_im)
windowNames.append("Grayscale")

bilateral_im = cv2.bilateralFilter(gray_im, 3, 50, 50)
images.append(bilateral_im)
windowNames.append("Bilateral Blur")

normalized_im2 = np.zeros(im_size)
normalized_im2 = cv2.normalize(bilateral_im,normalized_im2,0, 255, cv2.NORM_MINMAX)
images.append(normalized_im2)
windowNames.append("Normalized2")

thresh_im = cv2.adaptiveThreshold(bilateral_im,np.max(bilateral_im),cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,5,2)
#thresh_im = (255-thresh_im)
images.append(thresh_im)
windowNames.append("Adaptive Threshold - Bilateral")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

closed_im = cv2.morphologyEx(thresh_im, cv2.MORPH_ERODE, kernel)
images.append(closed_im)
windowNames.append("Closed Image")


ccl = cv2.connectedComponentsWithStats(closed_im)
print("Found %d labels" % ccl[0])
labels = ccl[1]

# Map component labels to hue val
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0

images.append(labeled_img)
windowNames.append("Labeled Image")

min_size = 50
good_labels=[]
centroids = []

cropped_images = []
center = (im_size[0]/2, im_size[1]/2)
minDist = 1000
minDistIndex = 0

for i in range(1, ccl[0]):
    if ccl[2][i, cv2.CC_STAT_AREA] > min_size:
        crop = np.zeros(im_size)
        crop.fill(i)
        crop = np.equal(ccl[1], crop).astype(np.uint8)
        crop = crop * 255
        
        left = ccl[2][i][0]
        top = ccl[2][i][1]
        width = ccl[2][i][2]
        height = ccl[2][i][3]
        
        bbox_im = crop[top:top+height, left:left+width]
        cropped_images.append(bbox_im)
        distance = tools.distance(ccl[3][i], center)
        if distance < minDist:
            minDist = distance
            minDistIndex = i
            eval_im = test_im[top:top+height, left:left+width]
            imshow('eval' + str(i), eval_im)
        #cv2.imshow("crop" + str(i), bbox_im)
        #cv2.waitKey(0)
        #cropped_images.append(crop)
        
images.append(eval_im)
windowNames.append("Evaluation Image")

'''
left = ccl[2][i][0]
top = ccl[2][i][1]
width = ccl[2][i][2]
height = ccl[2][i][3]
'''

windows = []
for i in range(len(images)):
    windows.append(cv2.namedWindow(windowNames[i], cv2.WINDOW_NORMAL))
    cv2.imshow(windowNames[i], images[i])

text = ''
i = 0


for im in cropped_images:
    #imshow('crop' + str(i), im)
    #cv2.waitKey(0)
    i = i + 1
    text = text + tess.image_to_string(im, lang='eng', config='--psm 10 -c tessedit_char_whitelist=0123456789')

fgh = eval_im
text = tess.image_to_string(fgh, lang='eng', config='-c tessedit_char_whitelist=0123456789')

print(type(text))
print(text)
cv2.waitKey(0)
