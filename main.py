import numpy as np
from PIL import Image
import os,glob
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
import loader
import model

#show functions
def show_segment(sg):
  sg = np.argmax(sg,axis=-1)
  plt.figure()
  plt.imshow(np.asarray(sg))
  plt.show()
def show_image(im):
    plt.figure()
    plt.imshow(im)
    plt.show()
def compare_TV(history,dir):
        # Setting Parameters
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        # 1) Accracy Plt
        plt.figure()
        plt.plot(epochs, acc, 'bo' ,label = 'training acc')
        plt.plot(epochs, val_acc, 'b' , label= 'validation acc')
        plt.title('Training and Validation acc')
        plt.legend()
        plt.savefig(dir+"/acc.png")

        # 2) Loss Plt
        plt.figure()
        plt.plot(epochs, loss, 'bo' ,label = 'training loss')
        plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.savefig(dir+"/loss.png")
#define the loss function
def my_loss(x,y):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(x,y))

def main():
    #load VOCdataset
    path_a , path_b = loader.generate_path("./VOCdataset/original","./VOCdataset/segmented")
    im = loader.image_generator(path_a)
    sg = loader.segment_generator(path_b)
    print("load is succesfully finished!")
    M = model.build_Segnet()
    save_dir ="Segnet"
    if(len(sys.argv)>1):
        save_dir=sys.argv[1]
    M.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["acc"])
    M.summary()
    history = M.fit(im[:2000],sg[:2000],batch_size = 4,validation_data = (im[2000:2500],sg[2000:2500]),epochs = 100)
    #history = M.fit(im[:2000],sg[:2000],batch_size = 4,epochs = 20)
    compare_TV(history,save_dir)
    for i in range(100):
        pred = M.predict(im[2500+i].reshape(1,64,64,3))
        sg = np.argmax(pred[0],axis=-1)
        plt.figure()
        plt.imshow(np.asarray(sg))
        plt.savefig("./"+save_dir+"/{}.png".format(i))
    for i in range(100):
        pred =  M.predict(im[i].reshape(1,64,64,3))
        sg = np.argmax(pred[0],axis=-1)
        plt.figure()
        plt.imshow(np.asarray(sg))
        plt.savefig("./"+save_dir+"/train{}.png".format(i))

if __name__=='__main__':
    main()
