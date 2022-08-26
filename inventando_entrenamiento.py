import numpy as np
import sys, argparse
import os
import json
import plotly
import plotly.express as px
import nsgrid_rsd as nsgrid
import multiprocessing
from multiprocessing import Pool
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
# For PointNet modules

#from models.pointnet import PointNet
#from utils.DataContainer import DataContainer as DC
from pointnet import PointNet
from DataContainer import DataContainer as DC



def main():
   
   args = get_args()      

   print('loading numpy data...')
   data = np.load(args.dataset)
   labels = np.load(args.labels)

   print('converting to DataContainer format...')
   dc = DC(data=data, labels=labels)
   
   net = PointNet(epochs=args.nepochs,
                   batch_size=32,
                   lr=args.learning_rate,
                   n_points=dc.train.data.shape[1],
                   n_classes=dc.train.labels.shape[-1],
                   n_input=dc.train.data.shape[-1],
                   #cutoff = args.cutoff,
                   #max_neigh = args.max_neigh, 
                   #n_samples = args.n_samples,
                   verbose=1,
                   weights_dir=args.weights)
                   
   print('train shape: ' + str(dc.train.data.shape))
   print('label shape: ' + str(dc.train.labels.shape))
   print('test shape: ' + str(dc.test.data.shape))
   print('label shape: ' + str(dc.test.labels.shape))
   
   acc = net.run(dc)

   print('final accuracy: ' + str(acc))
       


def get_args():
    #Parse Arguments
    parser = argparse.ArgumentParser(description='Run classification on specified dataset')
    parser.add_argument('--dataset', help='dataset to be used (numpy format)', type=str, required=True)
    parser.add_argument('--labels', help='labels corresponding to dataset (numpy format)', type=str, required=True)
    parser.add_argument('--weights', help='folder to save weights to', type=str, required=True)
    parser.add_argument('--learning_rate',help='learning rate',type=float, required=False, default=0.001)
    parser.add_argument('--nepochs', help='number of epochs for training', type=int, required=False,default=200)
#    parser.add_argument('--cutoff', help='neighbor cutoff for point clouds', type=float, required=True)
#    parser.add_argument('--max_neigh',help='max neighbors in point clouds',type=int,required=True)
#    parser.add_argument('--rate',help='rate',type=float,required=False, default=0.7)       
#    parser.add_argument('--n_samples', help='total number training samples per phase', type=int, required=False,default=100000)    
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
   with Pool() as pool:
      pool.map(main(), range(8))




