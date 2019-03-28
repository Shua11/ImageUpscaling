#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:23:44 2019

@author: thomas
"""
import numpy as np
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

x_train = [img_to_array(load_img('../HD pics/DIV2K_train_HR/{:04d}.png'.format(i))) for i in range(1,100)]


######################
input_img = layers.Input(shape=(None, None, 3))

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

######################

#autoencoder.fit(x_train, x_train, epochs=1, batch_size=500, shuffle=True)

#decoded_imgs = autoencoder.predict(x_test)

