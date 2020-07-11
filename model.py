import numpy as np
from PIL import Image
import os,glob
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
from keras.layers.merge import concatenate
from keras import backend as K
from keras.layers import Layer
class MaxPoolingWithArgmax2D(Layer):
    def __init__(self):
        super(MaxPoolingWithArgmax2D,self).__init__()
    def call(self,inputs):
        output,argmax = tf.nn.max_pool_with_argmax(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        argmax = K.cast(argmax,K.floatx())
        return [output,argmax]
    def compute_output_shape(self,input_shape):
        ratio = (1,2,2,1)
        output_shape = [dim//ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape,output_shape]
class MaxUnpooling2D(Layer):
    def __init__(self):
        super(MaxUnpooling2D,self).__init__()
    def call(self,inputs,output_shape = None):
        updates, mask = inputs[0],inputs[1]
        with tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0],input_shape[1]*2,input_shape[2]*2,input_shape[3])
            self.output_shape1 = output_shape
            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate([[input_shape[0]], [1 ], [1], [1]],axis=0)
            batch_range = K.reshape(tf.range(output_shape[0], dtype='int32'),shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret
    def compute_output_shape(self,input_shape):
        shape = input_shape[1]
        return (shape[0],shape[1]*2,shape[2]*2,shape[3])
def build_FCN():
  ffc = 32
  inputs = layers.Input(shape=(64,64,3))
  for i in range(2):
    x = layers.Conv2D(ffc,kernel_size=3,padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
  x = layers.MaxPooling2D((2,2))(x)
  for i in range(2):
    x = layers.Conv2D(ffc*2,kernel_size=3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
  x = layers.MaxPooling2D((2,2))(x)
  for i in range(3):
    x = layers.Conv2D(ffc*4,kernel_size=3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
  x = layers.MaxPooling2D((2,2))(x)
  for i in range(3):
    x = layers.Conv2D(ffc*8,kernel_size=3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
  x = layers.MaxPooling2D((2,2))(x)
  for i in range(3):
    x = layers.Conv2D(ffc*8,kernel_size=3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
  x = layers.UpSampling2D((2,2))(x)
  for i in range(3):
    x = layers.Conv2D(ffc*4,kernel_size=3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
  x = layers.UpSampling2D((2,2))(x)
  for i in range(3):
    x = layers.Conv2D(ffc*2,kernel_size=3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
  x = layers.UpSampling2D((2,2))(x)
  for i in range(2):
    x = layers.Conv2D(ffc*2,kernel_size=3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
  x = layers.UpSampling2D((2,2))(x)
  for i in range(2):
    x = layers.Conv2D(ffc,kernel_size=3,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
  x = layers.Conv2D(22,kernel_size=3,padding="same",activation="softmax")(x)
  return models.Model(inputs,x)

def build_Segnet():
    ffc = 32
    inputs = layers.Input(shape=(64,64,3))
    for i in range(2):
      x = layers.Conv2D(ffc,kernel_size=3,padding="same")(inputs)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
    x,x1 = MaxPoolingWithArgmax2D()(x)
    for i in range(2):
      x = layers.Conv2D(ffc*2,kernel_size=3,padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
    x,x2 = MaxPoolingWithArgmax2D()(x)
    for i in range(3):
      x = layers.Conv2D(ffc*4,kernel_size=3,padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
    x,x3 = MaxPoolingWithArgmax2D()(x)
    for i in range(3):
      x = layers.Conv2D(ffc*8,kernel_size=3,padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
    x,x4 = MaxPoolingWithArgmax2D()(x)
    for i in range(3):
      x = layers.Conv2D(ffc*8,kernel_size=3,padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
    x = layers.Dropout(rate = 0.5)(x)
    x = MaxUnpooling2D()([x,x4])
    for i in range(3):
      x = layers.Conv2D(ffc*4,kernel_size=3,padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
    x = MaxUnpooling2D()([x,x3])
    for i in range(3):
      x = layers.Conv2D(ffc*2,kernel_size=3,padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
    x = MaxUnpooling2D()([x,x2])
    for i in range(2):
      x = layers.Conv2D(ffc,kernel_size=3,padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
    x = MaxUnpooling2D()([x,x1])
    for i in range(2):
      x = layers.Conv2D(ffc,kernel_size=3,padding="same")(x)
      x = layers.BatchNormalization()(x)
      x = layers.ReLU()(x)
    x = layers.Conv2D(22,kernel_size=3,padding="same",activation="softmax")(x)
    return models.Model(inputs,x)
