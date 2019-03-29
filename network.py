#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:23:44 2019

@author: thomas
"""
import os
import numpy as np
import datetime 
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img


datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

images = [load_img('../HD pics/DIV2K_train_HR/{:04d}.png'.format(i+1)) for i in range(10)]
images2= [load_img('../HD pics/resized_training/{:04d}.png'.format(i+1)) for i in range(10)]
x_train = [img_to_array(x)/255 for x in images2]
y_train = [img_to_array(x)/255 for x in images]

######################
input_img = layers.Input(shape=(None, None, 3))

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
output = layers.Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x)

upscaler = models.Model(input_img, output)
upscaler.compile(optimizer='adadelta', loss='mean_absolute_error')

######################

for epoch in range(1):
    print("epoch #{}".format(epoch))
    for i in range(len(x_train)):
        upscaler.fit(np.array([x_train[i]]), np.array([y_train[i]]),
                     epochs=1, batch_size=5, shuffle=True,
                     verbose=0)
 
outdir = "../output_{}".format(datetime.datetime.now().strftime("%y%m%d"))
os.mkdir(outdir)
        
for i,x in enumerate(x_train):
    array_to_img(upscaler.predict(np.array([x]))[0]).save("{}/{:04d}.png".format(outdir,i+1))