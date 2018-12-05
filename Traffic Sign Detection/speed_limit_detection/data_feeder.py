import numpy as np
from easydict import EasyDict
import glob
import cv2
import csv

def data_feeder(label_indices,
                train_test_split= 0.7,
                input_image_size = (227,227),
                data_path = 'GTSRB\Final_Training\SpeedLimitSigns'
                ):
    
    num_classes = len(label_indices)
    
    im_paths = glob.glob(data_path + '/*/*.ppm')
    
    csv_paths = glob.glob(data_path + '/*/*.csv')
    fullcsvdata = []
    
    for path in csv_paths:
        
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter = ';')
            next(reader, None) #skip the header line in CSV
            onecsvdata = list(reader)
            for i in range(len(onecsvdata)):
                fullcsvdata.append(onecsvdata[i])
          
        
    class_indices = [el[7] for el in fullcsvdata]
    class_indices = list(map(int, class_indices)) #converting class indices from string to int
    
    num_training_examples = int(np.round(train_test_split*len(im_paths)))
    num_testing_examples = len(im_paths) - num_training_examples
    
    random_indices = np.arange(len(im_paths))
    np.random.shuffle(random_indices)

    training_indices = random_indices[:num_training_examples]
    testing_indices = random_indices[num_training_examples:]
    
    data = EasyDict()
    data.train = EasyDict()
    data.test = EasyDict()
    
        # Make empty arrays to hold data:
    data.train.X = np.zeros((num_training_examples, input_image_size[0], input_image_size[1], 3), 
                            dtype = 'float32')
    data.train.y = np.zeros((num_training_examples, num_classes), dtype = 'float32')

    data.test.X = np.zeros((num_testing_examples, input_image_size[0], input_image_size[1], 3), 
                            dtype = 'float32')
    data.test.y = np.zeros((num_testing_examples, num_classes), dtype = 'float32')


    for count, index in enumerate(training_indices):
        im = cv2.imread(im_paths[index])
        im = cv2.resize(im, (input_image_size[1], input_image_size[0]))
        data.train.X[count, :, :, :] = im
        
        class_index = class_indices[index] #pull class # from list
        data.train.y[count, class_index] = 1
    
    for count, index in enumerate(testing_indices):
        im = cv2.imread(im_paths[index])
        im = cv2.resize(im, (input_image_size[1], input_image_size[0]))
        data.test.X[count, :, :, :] = im
        
        class_index = class_indices[index] #class number is the eight element in the list
        data.test.y[count, class_index] = 1

    print('Loaded', str(len(training_indices)), 'training examples and ', 
          str(len(testing_indices)), 'testing examples. ')

    return data