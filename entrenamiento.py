import numpy as np
import sys, argparse
import os
import json
import multiprocessing
from multiprocessing import Pool
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import regularizers
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from red_puntos import PointNet
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'






def main():
   
   args = get_args()      
   
   #cargo data
   data = np.load(args.dataset)
   labels = np.load(args.labels)
   
   #la red
   
   net = PointNet(epochs=args.nepochs,
                   batch_size=args.batch_size,
                   lr=args.learning_rate,
                   n_classes=labels.shape[-1],
                   arg = args.arg,
                   rate = args.rate)
                   
   # roto data                
   point_cloud_samples = net.rotate_point_cloud(data)
   a,b,c = point_cloud_samples.shape
   point_cloud_samples = point_cloud_samples.reshape(a,b,c,-1)     

             
   # split data
   full_train_samples,test_samples,full_train_labels,test_labels = train_test_split(point_cloud_samples, labels, test_size=0.2, random_state=13)
   train_samples,valid_samples,train_labels,valid_labels = train_test_split(full_train_samples, full_train_labels, test_size=0.2, random_state=13)

                   
   print('train shape: ' + str(train_samples.shape))
   print('label shape: ' + str(train_labels.shape))
   print('valid shape: ' + str(valid_samples.shape))
   print('label shape: ' + str(valid_labels.shape))   
   
   

   #red = net.defino_red(1e-5,0.3,n_classes,train_samples[1].shape)
   
   #red.summary()
   
   #Entreno                        
   PointNet_train = net.entreno_red(train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, args.batch_size, args.nepochs)
   
   return

   
def get_args():
    #Parse Arguments
    parser = argparse.ArgumentParser(description='Run classification on specified dataset')
    parser.add_argument('--dataset', help='dataset to be used (numpy format)', type=str, required=True)
    parser.add_argument('--labels', help='labels corresponding to dataset (numpy format)', type=str, required=True)
    parser.add_argument('--learning_rate',help='learning rate',type=float, required=False, default=0.001)
    parser.add_argument('--nepochs', help='number of epochs for training', type=int, required=False, default=150)
    parser.add_argument('--batch_size', help='size of batch for training', type=int, required=False, default=32)
    parser.add_argument('--arg',help='argument',type=float, required=False, default=1e-5)
    parser.add_argument('--rate',help='rate of dropout',type=float, required=False, default=0.7)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
   with Pool() as pool:
      pool.map(main(), range(8))




