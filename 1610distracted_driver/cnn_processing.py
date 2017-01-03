import cPickle as pickle
import numpy as np
import tensorflow as tf
import sys
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


## load train, test and validation dataset ##
f = open('./img_train.p', 'wb')   
X_train = pickle.load(f)          
f.close()  
f = open('./img_train_label.p', 'wb')  
y_train = pickle.load(f)
f.close()

f = open('./img_test.p', 'wb')   
X_test = pickle.load(f)         
f.close()  
f = open('./img_test_label.p', 'wb')  
y_test = pickle.load(f)
f.close()

f = open('./img_valid.p', 'wb')   
X_valid = pickle.load(f)         
f.close()  
f = open('./img_valid_label.p', 'wb')  
y_valid = pickle.load(f)
f.close()


# input image dimensions 
img_rows, img_cols = 50,50
# number of convolutional filters to use
nb_filters = 32
## size of pooling area for max pooling
pool_size = (3, 3)
# convolution kernel size
kernel_size = (5, 5)
nb_classes = 10
input_shape = (img_rows, img_cols,1) ##this is tf backend, for theano (1,50,50)

## CNN model architecture ##
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
conv1=model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])


model.summary() ## checking model architecture ##

## Fit the model ##
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                    nb_epoch=50, batch_size=32,verbose=1)


## checking loss ##
predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
score = log_loss(y_valid, predictions_valid)
print('Score log_loss: ', score)

## predict on test set ##

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


## plotting accuracy curves for train and validation ##
# list all data in history
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim(0,1.05)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
#plt.show()
plt.savefig('./model_accuracy1.png',dpi=300)



## plotting loss curves for train and validation  ##
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylim(-0.05,3)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
#plt.show()
plt.savefig('./model_loss1.png',dpi=300)

## classification report for test data ##
y_pred = np.argmax(predictions_test, axis=1)
y_test1 = np.argmax(y_test,axis=1)
print(classification_report(y_test1, y_pred))

## Confusion matrix for test data ##
a = range(10)
cm = confusion_matrix(y_test1, y_pred)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + a)
ax.set_yticklabels([''] + a)
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.show()
plt.savefig('./confusion_matrix1.png',dpi=300)


## reference: http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
