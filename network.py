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

images = [load_img('../HD pics/DIV2K_train_HR/{:04d}.png'.format(i+1)) for i in range(100)]
y_train = np.array([img_to_array(x)[:768, :768, :]/255 for x in images])
x_train = np.array([img_to_array(x.resize((384,384)))[:384, :384, :]/255 for x in images])

######################
input_img = layers.Input(shape=(None, None, 3))

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (5, 5), activation='relu', padding='same')(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

######################

autoencoder.fit(x_train, y_train, epochs=1, batch_size=5, shuffle=True)

#decoded_imgs = autoencoder.predict(x_test)

