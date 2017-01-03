import os, glob, math, cv2, time
import numpy as np
from skimage import io
from joblib import Parallel, delayed
import cPickle as pickle
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
import keras
from keras.utils import np_utils

img_size = 50
sz = (img_size, img_size) ## image shape

nprocs = 4 ## number of processor will be used parallel processing

## image resize function
def resize_image(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, sz).transpose((2,0,1)).astype('float32') / 255.0
    return img

start = time.time()
X_train = []
Y_train = []

## reading image in parallel process ##
for j in range(10):
    print('Load folder c{}'.format(j))
    path = os.path.join('./input/train', 'c' + str(j), '*.jpg')
    files = glob.glob(path)
    X_train.extend(Parallel(n_jobs=nprocs)(delayed(resize_image)(im_file) for im_file in files))
    Y_train.extend([j]*len(files))
    
end = time.time() - start
print("Time: %.2f seconds" % end)

## converting to numpy array from list##
data= np.array(X_train) 
label = np.array(Y_train)

##saving data ##
f = open('./img_data.p', 'wb')   
pickle.dump(data, f)          
f.close()  
f = open('./img_label.p', 'wb')  
pickle.dump(label, f)          # dump data to f
f.close()


##stratified sampling to train, test and validation set
s = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=0)
for train_index, test_index in s.split(data,label):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train1, X_test = data[train_index], data[test_index]
    y_train1, y_test = label[train_index], label[test_index]
    
s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in s.split(X_train1,y_train1):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_valid = X_train1[train_index], X_train1[test_index]
    y_train, y_valid = y_train1[train_index], y_train1[test_index]

## data reformat ##
image_size=50
num_labels=10
def reformat(dataset1):
    dataset1 = dataset1.reshape(dataset1.shape[0], image_size, image_size,1).astype(np.float32)
    #dataset2 = dataset2.reshape(dataset2.shape[0], 1, image_size, image_size)
    #dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    #labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    #labels = labels.reshape((-1,10))
    return dataset1
X_train = reformat(X_train)
X_valid = reformat(X_valid)
X_test = reformat(X_test)

## one hot encoding ##
y_train = np_utils.to_categorical(y_train, num_labels)
y_test = np_utils.to_categorical(y_test, num_labels)
y_valid = np_utils.to_categorical(y_valid, num_labels)

## saving train,test and validation data ##
f = open('./img_train.p', 'wb')   
pickle.dump(X_train, f)          
f.close()  
f = open('./img_train_label.p', 'wb')  
pickle.dump(y_train, f)
f.close()

f = open('./img_test.p', 'wb')   
pickle.dump(X_test, f)          
f.close()  
f = open('./img_test_label.p', 'wb')  
pickle.dump(y_test, f)
f.close()

f = open('./img_valid.p', 'wb')   
pickle.dump(X_valid, f)          
f.close()  
f = open('./img_valid_label.p', 'wb')  
pickle.dump(y_valid, f)
f.close()






##ref: https://www.kaggle.com/inoryy/state-farm-distracted-driver-detection/fast-image-pre-process-in-parallel/comments
