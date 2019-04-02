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

class croppedSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.cropsize = 300
        self.scale = 4

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
        if w < 300 or h < 300:
            return self.getRandomCrop()
        x = np.random.randint(0, w-299)
        y = np.random.randint(0, h-299)
        
        return (self.x[i][x: x + self.cropsize, y: y + self.cropsize, :],
                self.y[i][x * self.scale: (x + self.cropsize) * self.scale,
                          y * self.scale: (y + self.cropsize) * self.scale, :])

images = [load_img('../HD pics/DIV2K_train_HR/{:04d}.png'.format(i+1)) for i in range(22, 100)]
images2= [load_img('../HD pics/resized_training/{:04d}.png'.format(i+1)) for i in range(22, 100)]
x_train = [img_to_array(x)/255 for x in images2]
y_train = [img_to_array(x)/255 for x in images]

######################
input_img = layers.Input(shape=(None, None, 3))

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(x)
output = layers.Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x)

upscaler = models.Model(input_img, output)
upscaler.compile(optimizer='adadelta', loss='mean_absolute_error')

###################### Train

train_generator = croppedSequence(x_train, y_train, 5)

upscaler.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=10)
"""
for epoch in range(1):
    print("epoch #{}".format(epoch+1))
    start = datetime.now()
    for i in range(len(x_train)):
        print("  image #{}".format(i+1))
        upscaler.fit(np.array([x_train[i]]), np.array([y_train[i]]),
                     epochs=1, batch_size=5, shuffle=True,
                     verbose=0)
    print("time: #{}".format(datetime.now() - start))
 """
##################### Output data

outdir = "../output_{}".format(datetime.now().strftime("%y%m%d%H%M"))
os.mkdir(outdir)
        
for i,x in enumerate(x_train):
    array_to_img(upscaler.predict(np.array([x]))[0]).save("{}/{:04d}.png".format(outdir, i+1))