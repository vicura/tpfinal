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
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import regularizers
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


    # Function modified from https://github.com/charlesq34/pointnet/blob/master/provider.py
def rotate_point_cloud(batch_data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            angles = np.random.uniform(size=(3)) * 2 * np.pi
            cosval = np.cos(angles)
            sinval = np.sin(angles)

            x_rot_mat = np.array([[1, 0, 0],
                                  [0, cosval[0], -sinval[0]],
                                  [0, sinval[0], cosval[0]]])

            y_rot_mat = np.array([[cosval[1], 0, sinval[1]],
                                  [0, 1, 0],
                                  [-sinval[1], 0, cosval[1]]])

            z_rot_mat = np.array([[cosval[2], -sinval[2], 0],
                                  [sinval[2], cosval[2], 0],
                                  [0, 0, 1]])

            # Overall rotation calculated from x,y,z -->
            # order matters bc matmult not commutative 
            overall_rot = np.dot(z_rot_mat,np.dot(y_rot_mat,x_rot_mat))
            # Transposes bc overall_rot operates on col. vec [[x,y,z]]
            rotated_pc = np.dot(overall_rot,batch_data[k,:,:3].T).T
            rotated_data[k] = np.concatenate((rotated_pc, batch_data[k,:,3:]), axis=1)

        return rotated_data





def main():
   
   args = get_args()      
   
   #cargo data
   data = np.load(args.dataset)
   labels = np.load(args.labels)
   
   # roto data                
   point_cloud_samples = rotate_point_cloud(data)
   a,b,c = point_cloud_samples.shape
   point_cloud_samples = point_cloud_samples.reshape(a,b,c,-1)     

             
   # split data
   full_train_samples,test_samples,full_train_labels,test_labels = train_test_split(point_cloud_samples, labels, test_size=0.2, random_state=13)
   train_samples,valid_samples,train_labels,valid_labels = train_test_split(full_train_samples, full_train_labels, test_size=0.2, random_state=13)

                   
   print('train shape: ' + str(train_samples.shape))
   print('label shape: ' + str(train_labels.shape))
   print('valid shape: ' + str(valid_samples.shape))
   print('label shape: ' + str(valid_labels.shape))   
   
   os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
   os.environ["CUDA_VISIBLE_DEVICES"]="1"
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   
   #la red
   PointNet = Sequential()
   PointNet.add(Conv2D(64, kernel_size=(1,3), activation= 'relu',
                             padding='same', 
                            #activity_regularizer=keras.regularizers.l2(1e-5),
                            kernel_regularizer=keras.regularizers.l2(1e-5),
                            input_shape = train_samples[1].shape ))
   PointNet.add(Conv2D(64, kernel_size=(1,1), activation= 'relu',
                            #is_training=is_training,
                            #activity_regularizer=keras.regularizers.l2(1e-5),
                            kernel_regularizer=keras.regularizers.l2(1e-5),
                            padding='same'))
   PointNet.add(Conv2D(64, kernel_size=(1,1), activation= 'relu',
                            #is_training=is_training,
                            #activity_regularizer=keras.regularizers.l2(1e-5),
                            kernel_regularizer=keras.regularizers.l2(1e-5),
                            padding='same'))
   PointNet.add(Conv2D(128, kernel_size=(1,1), activation= 'relu',
                           # is_training=is_training,
                            #activity_regularizer=keras.regularizers.l2(1e-5),
                            kernel_regularizer=keras.regularizers.l2(1e-5),
                            padding='same'))
   PointNet.add(Conv2D(1024, kernel_size=(1,1), activation= 'relu',
                            #is_training=is_training,
                            #activity_regularizer=keras.regularizers.l2(1e-5),
                            kernel_regularizer=keras.regularizers.l2(1e-5),
                            padding='same'))

        # Symmetric function: max pooling
   PointNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),
                                 padding='same'))
        
        # MLP on global point cloud vector
        
   PointNet.add(Flatten())
   PointNet.add(Dense(512, activation='relu',#is_training=is_training,
                           #activity_regularizer=keras.regularizers.l2(1e-5),
                           kernel_regularizer=keras.regularizers.l2(1e-5)))
   PointNet.add(Dense(256, activation= 'relu',#is_training=is_training,
                           #activity_regularizer=keras.regularizers.l2(1e-5),
                           kernel_regularizer=keras.regularizers.l2(1e-5)))
   PointNet.add(Dropout(rate=0.3))
   PointNet.add(Dense(args.n_classes, activation='softmax'))

   PointNet.compile(loss=keras.losses.categorical_crossentropy, optimizer= tf.keras.optimizers.Adam(),metrics=['accuracy'])

   PointNet.summary()          

          
   # model callback

   early_stop = EarlyStopping(monitor='val_accuracy',
                           patience=10,
                           restore_best_weights=True,
                           mode='max')

   #Entreno                        
   PointNet_train = PointNet.fit(train_samples, train_labels, batch_size = args.batch_size, epochs = args.nepochs, verbose= 1, 
        callbacks=[early_stop], validation_data=(valid_samples, valid_labels))
   
   #Evaluo
   mse_test = PointNet.evaluate(test_samples, test_labels)

   print('mse_test: ' + str(mse_test))
       
   for key in PointNet_train.history.keys():
       print(key)
    
   accuracy = PointNet_train.history['accuracy']
   val_accuracy = PointNet_train.history['val_accuracy']
   loss = PointNet_train.history['loss']
   val_loss = PointNet_train.history['val_loss']
   epochs = range(len(accuracy))

   plt.figure(figsize=(10, 5))
   plt.subplot(1, 2, 1)
   plt.plot(epochs, accuracy, label='Training accuracy')
   plt.plot(epochs, val_accuracy, label='Validation accuracy')
   plt.title('Training and validation accuracy')
   plt.legend()

   plt.subplot(1, 2, 2)
   plt.plot(epochs, loss, label='Training loss')
   plt.plot(epochs, val_loss, label='Validation loss')
   plt.title('Training and validation loss')
   plt.legend()
   plt.show()
   
   # Plot non-normalized confusion matrix
   titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
            ]
   for title, normalize in titles_options:
      disp = ConfusionMatrixDisplay.from_estimator(
        PointNet_train,
        test_samples,
        test_labels,
        #display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
           )
      disp.ax_.set_title(title)

      print(title)
      print(disp.confusion_matrix)

   plt.show()
   
   
def get_args():
    #Parse Arguments
    parser = argparse.ArgumentParser(description='Run classification on specified dataset')
    parser.add_argument('--dataset', help='dataset to be used (numpy format)', type=str, required=True)
    parser.add_argument('--labels', help='labels corresponding to dataset (numpy format)', type=str, required=True)
    parser.add_argument('--n_classes', help='number of classes', type=int, required=True)
    parser.add_argument('--learning_rate',help='learning rate',type=float, required=False, default=0.001)
    parser.add_argument('--nepochs', help='number of epochs for training', type=int, required=False, default=150)
    parser.add_argument('--batch_size', help='size of batch for training', type=int, required=False, default=32)
   # parser.add_argument('--batch_size', help='size of batch for training', type=int, required=False, default=32)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
   with Pool() as pool:
      pool.map(main(), range(8))




