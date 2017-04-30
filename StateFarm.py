import math, os, sys
import pickle
from glob import glob
import numpy as np
from numpy.random import random, permutation, randn, normal
from matplotlib import pyplot as plt
#%matplotlib inline
import PIL
from PIL import Image
import bcolz
from shutil import copyfile
from shutil import move

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_crossentropy
from keras.regularizers import l2,l1


current_dir = os.getcwd()
path = current_dir + '/data/'
test_path = path +'test/'
results_path = path + 'results/'
train_path = path+ 'train/'
valid_path = path+ 'valid/'

batch_size =1;

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, 
                batch_size=batch_size, target_size=(224,224), class_mode='categorical'):
    return gen.flow_from_directory(path+dirname, target_size, 
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
		
def get_classes(path):
    batches = get_batches('train', shuffle=False, batch_size=1)
    val_batches = get_batches('valid', shuffle=False, batch_size=1)
    #test_batches = get_batches('test', shuffle=False, batch_size=1)
    return (val_batches.classes, batches.classes,to_categorical(val_batches.classes),
            to_categorical(batches.classes),val_batches.filenames, batches.filenames)
           # test_batches.filenames)

def get_data(path, target_size = (224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range (len(batches.classes))])

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()	
def load_array(fname):
    return bcolz.open(fname)[:]
	
# Roll into pixcel matrix
#train_data = get_data('train')
#valid_data = get_data('valid')


#save_array(path+'results/train_data.dat',  train_data)
#save_array(path+'results/valid_data.dat',  valid_data)

# (val_classes, trn_classes, val_labels, trn_labels, val_filenames, filenames,
    # test_filename) = get_classes(path)
(valid_classes, train_classes, valid_labels, train_labels, valid_filenames, train_filenames) = get_classes(path)
valid_data = load_array(path+'results/valid_data.dat')
train_data = load_array(path+'results/train_data.dat')


model = Sequential([
        BatchNormalization(axis=-1, input_shape=(224,224,3)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
	
model.compile(Adam(lr=10e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
batch_size=32
Ex1_history = model.fit(train_data,train_labels, batch_size=batch_size, epochs=10, validation_data =(valid_data,valid_labels))
                
				

plt.figure(1)
plt.subplot(211)
plt.plot(Ex1_history.history['acc'])
plt.plot(Ex1_history.history['val_acc'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(212)
plt.plot(Ex1_history.history['loss'])
plt.plot(Ex1_history.history['val_loss'])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoches')
plt.legend(['train', 'validation'], loc='upper left')

plt.savefig(results_path+'/train_history/Exp1_train_history_1.png', bbox_inches='tight')
plt.show()