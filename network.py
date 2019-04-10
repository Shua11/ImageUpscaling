#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:23:44 2019

@author: thomas
"""
import os
import numpy as np
from datetime import datetime 
from keras import layers, models
from keras.utils import Sequence
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import tensorflow as tf
import keras.backend as K

def ssim_metric(y_true, y_pred):
    # source: https://gist.github.com/Dref360/a48feaecfdb9e0609c6a02590fd1f91b
    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
    patches_true = tf.extract_image_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    patches_pred = tf.extract_image_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

    u_true = K.mean(patches_true, axis=3)
    u_pred = K.mean(patches_pred, axis=3)
    var_true = K.var(patches_true, axis=3)
    var_pred = K.var(patches_pred, axis=3)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom
    ssim = tf.where(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    return ssim

def sum_metric(y_true, y_pred):
    return ssim_metric(y_true, y_pred) + K.mean(K.abs(y_true - y_pred)) * 3

####### GPU Session Settings ####################################
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))
#################################################################

class croppedSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, cropsize = 300, scale = 2):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.cropsize = cropsize
        self.scale = scale
        #for i in range(len(self.x)):
            #assert(self.x[0].shape[0] * scale == self.y.shape[0]), "Incompatible image shapes"

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        xdata = np.zeros((self.batch_size,  self.cropsize,  self.cropsize, 3))
        ydata = np.zeros((self.batch_size,  self.cropsize * self.scale,  self.cropsize * self.scale, 3))
        for i in range(self.batch_size):
            xdata[i], ydata[i] = self.getRandomCrop()
        return xdata, ydata
    
    def getRandomCrop(self):
        i = np.random.randint(0, len(self.x))
        w,h,_ = self.x[i].shape
        if w < self.cropsize or h < self.cropsize:
            return self.getRandomCrop()
        x = np.random.randint(0, w-self.cropsize)
        y = np.random.randint(0, h-self.cropsize)
        
        return (self.x[i][x: x + self.cropsize, y: y + self.cropsize, :],
                self.y[i][x * self.scale: (x + self.cropsize) * self.scale,
                          y * self.scale: (y + self.cropsize) * self.scale, :])

##### Load images for training/testing #####
images = [load_img('../HD pics/DIV2K_train_HR/{:04d}.png'.format(i+1)) for i in range(22, 100)]
images2= [load_img('../HD pics/resized_training/{:04d}.png'.format(i+1)) for i in range(22, 100)]
x_train = [img_to_array(x)/255 for x in images2]
y_train = [img_to_array(x)/255 for x in images]


##### Start a new training model or load a pre-existing one? #####
is_Loading_Exisiting_Model = input("Using pre-existing model (type \"y\" or \"n\")? ")
path_to_model = ""
upscaler = None

## Ensure user types in valid option ##
while is_Loading_Exisiting_Model != "y" and is_Loading_Exisiting_Model != "n":
    is_Loading_Exisiting_Model = input("Please type y/n: ")

## Load previously created model ##
if is_Loading_Exisiting_Model == "y":
    path_to_model = input("Type path to model: ")
    upscaler = models.load_model(path_to_model)

else:
    input_img = layers.Input(shape=(None, None, 3))

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(x)
    output = layers.Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x)

    upscaler = models.Model(input_img, output)
    upscaler.compile(optimizer='adadelta', loss='mean_absolute_error')

###################### Train

train_generator = croppedSequence(x_train, y_train, 10)

upscaler.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=10)

##################### Output data

doesSavePictures = input("Save pictures (y/n)? ")
while doesSavePictures != "y" and doesSavePictures != "n":
    doesSavePictures = input("Please state \'y\' or \'n\': ")

if doesSavePictures == "y":
    outdir = "../output_{}".format(datetime.now().strftime("%y%m%d%H%M"))
    os.mkdir(outdir)
            
    for i,x in enumerate(x_train):
        array_to_img(upscaler.predict(np.array([x]))[0]).save("{}/{:04d}.png".format(outdir, i+1))

doesSaveModel = input("Save the current model (y/n)? ")
while doesSaveModel != "y" and doesSaveModel != "n":
    doesSaveModel = input("y/s please: ")

if doesSaveModel == "y":
    saveFilePath = input("Please specify filepath to save model: ")
    upscaler.save(saveFilePath)

