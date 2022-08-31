#/usr/bin/python

import numpy as np
from sklearn import preprocessing
import sys, argparse
import tf_util
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from keras import regularizers
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report



class PointNet:
    def __init__(self, lr=0.001, epochs=75,  \
        batch_size=16, disp_step=1, input_shape=(250,3,1), \
        rate = 0.3, arg = 1e-5,  \
        n_classes=3):

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.display_step = disp_step
        self.input_shape = input_shape
        self.rate = rate
        self.arg = arg
        self.n_classes = n_classes


    # Model definition for pointnet
    def pointnet(self, arg, rate, n_classes, input_shape):
        """ Classification PointNet, input is BxNx3, output Bxn where n is num classes """

        
        if arg != self.arg:
            self.arg = arg
        else:
            arg = self.arg
            
        if rate != self.rate:
            self.rate = rate
        else:
            rate = self.rate
        
        if n_classes != self.n_classes:
            self.n_classes = n_classes
        else:
            n_classes = self.n_classes
            
        if input_shape != self.input_shape:
            self.input_shape = input_shape
        else:
            input_shape = self.input_shape  
        # Point functions (MLP implemented as conv2d)

        PointNet = Sequential()
        PointNet.add(Conv2D(64, kernel_size=(1,3), activation= 'relu',
                             padding='same', 
                            #activity_regularizer=keras.regularizers.l2(1e-5),
                            kernel_regularizer=keras.regularizers.l2(self.arg),
                            input_shape =self.input_shape))
        PointNet.add(Conv2D(64, kernel_size=(1,1), activation= 'relu',
                            #is_training=is_training,
                            #activity_regularizer=keras.regularizers.l2(1e-5),
                            kernel_regularizer=keras.regularizers.l2(self.arg),
                            padding='same'))
        PointNet.add(Conv2D(64, kernel_size=(1,1), activation= 'relu',
                            #is_training=is_training,
                            #activity_regularizer=keras.regularizers.l2(1e-5),
                            kernel_regularizer=keras.regularizers.l2(self.arg),
                            padding='same'))
        PointNet.add(Conv2D(128, kernel_size=(1,1), activation= 'relu',
                           # is_training=is_training,
                            #activity_regularizer=keras.regularizers.l2(1e-5),
                            kernel_regularizer=keras.regularizers.l2(self.arg),
                            padding='same'))
        PointNet.add(Conv2D(1024, kernel_size=(1,1), activation= 'relu',
                            #is_training=is_training,
                            #activity_regularizer=keras.regularizers.l2(1e-5),
                            kernel_regularizer=keras.regularizers.l2(self.arg),
                            padding='same'))

        # Symmetric function: max pooling
        PointNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),
                                 padding='same'))
        
        # MLP on global point cloud vector
        
        PointNet.add(Flatten())
        PointNet.add(Dense(512, activation='relu',#is_training=is_training,
                           #activity_regularizer=keras.regularizers.l2(1e-5),
                           kernel_regularizer=keras.regularizers.l2(self.arg)))
        PointNet.add(Dense(256, activation= 'relu',#is_training=is_training,
                           #activity_regularizer=keras.regularizers.l2(1e-5),
                           kernel_regularizer=keras.regularizers.l2(self.arg)))
        PointNet.add(Dropout(rate=self.rate))
        PointNet.add(Dense(self.n_classes, activation='softmax'))

        PointNet.compile(loss=keras.losses.categorical_crossentropy, optimizer= tf.keras.optimizers.Adam(),metrics=['accuracy'])

        
        return PointNet

    
    def defino_red(self,arg,rate,n_classes,input_shape):
       
       red = self.pointnet(arg,rate,n_classes,input_shape)
       
       return red
    
    
    def entreno_red(self,train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, batch_size, epochs): 
           
           # model callback

        early_stop = EarlyStopping(monitor='val_accuracy',
                           patience=2,
                           restore_best_weights=True,
                           mode='max')
                           
        red = self.defino_red(self.arg,self.rate,train_labels.shape[-1],train_samples[1].shape)
        
        PointNet_train = red.fit(train_samples, train_labels, batch_size = self.batch_size, epochs = self.epochs, verbose= 1, 
            callbacks=[early_stop], validation_data=(valid_samples, valid_labels))
           

          
        #Evaluo
        mse_test = red.evaluate(test_samples, test_labels)

        print('mse_test: ' + str(mse_test))
        

        #Plot train history
        
        #for key in PointNet_train.history.keys():
           #print(key)
    
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
        plt.savefig('train_history.png')
        #plt.show()
   
        # Plot confusion matrix
        
        predictions = red.predict(test_samples)
        predicted_labels = np.argmax(predictions, axis=1)       
        test_labels2 =  np.argmax(test_labels, axis=1)
        #print(predicted_labels.shape, test_labels2.shape)
        #print(predicted_labels, test_labels2)
        
        print("Confusion matrix")
        confusion_matrices = multilabel_confusion_matrix(test_labels2, predicted_labels)
        plt.figure(figsize=(12, 6))
        for n,confusion_matrix in enumerate(confusion_matrices):
           n = n+1
           
           plt.subplot(1,3,n)
           disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=test_labels)
           disp.plot()
           plt.show()
                           
        plt.savefig('cm.png')
        #plt.show()
        
        # Classification report 
        print(classification_report(test_labels2, predicted_labels))
        
        
        red.save_weights("pointnet_weights.ckpt")         
        return 
        
        
    def predigo_con_red(self, arg, rate, n_classes, input_shape, samples, step):
       
       red = self.defino_red(self.arg,self.rate,train_labels.shape[-1],train_samples[1].shape)     
       
       red = red.load_weights("pointnet_weights.ckpt")
       
       red.summary()
       
       pred = red.predict(samples, step)
       
       return pred
               
        
    # Function modified from https://github.com/charlesq34/pointnet/blob/master/provider.py
    def rotate_point_cloud(self, batch_data):
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





