# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:22:43 2020
@author: Tobias
"""
import tensorflow as tf
from tensorflow import reshape
from tensorflow.keras import models
from tensorflow.keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt

def grad_cam(model, layer_name, img_location):
    im = img_location
    
    image = cv2.imread(im, cv2.IMREAD_UNCHANGED)
    dim = (128,128)
    
    img_tensor = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    #img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_BGR2RGB)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor/255.
    
    def preprocess(img):
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img,axis=0)
        return img
    
    image_1 = preprocess(img_tensor)
    
    predict = model.predict(image_1)
    target_class = np.argmax(predict[0])
    print("Target Class = %d"%target_class)
    
    last_conv = model.get_layer(layer_name)
    heatmap_model = models.Model([model.inputs], [last_conv.output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = heatmap_model(image_1)
        loss = predictions[:, np.argmax(predictions[0])]
        preds = model(image_1)
        grads = tape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads,axis=(0,1,2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    
    heatmap = np.maximum(heatmap,0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    
    heatmap = reshape(heatmap, (8,8), 3)
    plt.imshow(heatmap)
    
    #Restore image dimensions
    #image[:,:, 1].shape[1] -->> Corresponds to the y-axis of the img dimensions,
    #image[:,:, 1].shape[0] -->> Corresponds to the x-axis of the img dimensions
    heatmap = np.expand_dims(heatmap,axis=-1)
    upsample = cv2.resize(heatmap, (image[:,:, 1].shape[1],image[:,:, 1].shape[0]), 3)
    
    plt.imshow(image)
    plt.imshow(upsample,alpha=0.5)
    plt.show()