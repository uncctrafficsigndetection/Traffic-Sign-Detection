import numpy as np
from easydict import EasyDict
import glob
import cv2
import csv
import pytesseract as tess
from docutils.nodes import Labeled

def Modify(input_Image, size = (100, 100)):              
    test_im = cv2.resize(input_Image, size)
    images.append(test_im)
    windowNames.append("Normal")
    
    gray_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
    images.append(gray_im)
    windowNames.append("Grayscale")
    
    bilateral_im = cv2.bilateralFilter(gray_im, 3, 50, 50)
    images.append(bilateral_im)
    windowNames.append("Bilateral Blur")
    
    thresh_im = cv2.adaptiveThreshold(bilateral_im,np.max(test_im),cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,5,2)
    thresh_im = (255-thresh_im)
    images.append(thresh_im)
    windowNames.append("Adaptive Threshold - Bilateral")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    closed_im = cv2.morphologyEx(thresh_im, cv2.MORPH_OPEN, kernel)
    images.append(closed_im)
    windowNames.append("Closed Image")
    
    
    ccl = cv2.connectedComponentsWithStats(closed_im)
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
    
    min_im = closed_im
    for i in range(ccl[0]):
        if ccl[2][i, cv2.CC_STAT_AREA] < min_size:
            min_im[ccl[1] == i] = 0
        else:
            good_labels.append(i)
    
    images.append(min_im)
    windowNames.append("Min Removed Image")
    
    print(ccl[1])
    uniqueVals = np.unique(min_im)
    print(uniqueVals)
    print(good_labels)
    
    good_stats = [ccl[2][i] for i in good_labels]
    
    #left = ccl[2][good_labels[1]][0]
    left = good_stats[1][0]
    top = good_stats[1][1]
    width = good_stats[1][2]
    height = good_stats[1][3]
    
    cropped_im = labeled_img[top:top+height, left:left+width]
    images.append(cropped_im)
    windowNames.append("Cropped Image")
    
    print(cropped_im)
    mask = cropped_im & 12
    print('mask:')
    print(mask)
    images.append(mask)
    windowNames.append("Mask")
    
    
    
    windows = []
    for i in range(len(images)):
        windows.append(cv2.namedWindow(windowNames[i], cv2.WINDOW_NORMAL))
        cv2.imshow(windowNames[i], images[i])
    
    text = tess.image_to_string(thresh_im, lang='eng', config='tessedit_char_whitelist=0123456789')
    
    print(type(text))
    print(text)
    cv2.waitKey(0)
