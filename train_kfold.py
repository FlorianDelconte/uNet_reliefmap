import numpy as np
import os
import sys
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
#import skimage.io as io
#import skimage.transform as trans
#import skimage.transform as trans
import numpy as np

#from matplotlib import pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
from random_eraser import get_random_eraser  # added
import model
import data
import math
import tensorflow as tf
import glob
import cv2
#Parameters Input
numChannels = model.channels
height  = model.height
width   = model.width
batch_size = model.batch_size
epochs =model.epochs

#Folders crossvalidation
save_dir        = os.path.join(os.getcwd())

input_folder    = 'input'
label_folder    = 'output'

# Folders for k-fold with k=5
kfolder=["k0","k1","k2","k3","k4"]

#Augmentated Data Operation
data_gen_args = dict(rescale = 1.0 / 255,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='reflect',
                    preprocessing_function=get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                    v_l=0, v_h=0, pixel_level=False))



############loop for k-fold training
for k in range(len(kfolder)):

    modelname       ="k"+str(k)
    fullname        = "".join([save_dir, '/'])+modelname+".hdf5"


    train_dir       = os.path.join(os.getcwd(), "dbRelief/thumbnails/","kfold",kfolder[k], 'train')
    valid_dir       = os.path.join(os.getcwd(), "dbRelief/thumbnails/","kfold",kfolder[k], 'valid')
    steps_per_epoch=(math.ceil((len([name for name in os.listdir(train_dir+"/input") if os.path.join(train_dir+"/input", name)]))/batch_size))*3
    validation_steps=(math.ceil((len([name for name in os.listdir(valid_dir+"/input") if os.path.join(valid_dir+"/input", name)]))/batch_size))*3
    # Create network
    __generators    = data.trainset_generator(model.batch_size, train_dir, input_folder, label_folder,
                                     data_gen_args, save_to_dir=None)#"../crossValidationv3/data4_1st/visu_augmented_data"

    __validator     = data.trainset_generator(model.batch_size, valid_dir, input_folder, label_folder,
                                              data_gen_args, save_to_dir=None)#"../crossValidationv3/data4_1st/visu_augmented_data"

    model_checkpoint= ModelCheckpoint(fullname,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True)

    net = model.unet()
    net.summary()

    try:
        net.load_weights(fullname)
        print("model already exist !!!")

    except IOError:
        pass

    
    print("######################################################################################")
    print(modelname)
    print(train_dir)
    print(valid_dir)
    print("######################################################################################")
    history=net.fit(__generators,
            steps_per_epoch   = steps_per_epoch,
            validation_data   = __validator,
            validation_steps  = validation_steps,
            epochs=epochs,
            callbacks=[model_checkpoint])
    print(history.history)

