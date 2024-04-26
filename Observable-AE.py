# Observable-AE.py
# 2023 Kai Fukami (UCLA, kfukami1@g.ucla.edu)

## Authors:
# Kai Fukami and Kunihiko Taira 
## We provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citation, please use the reference below:
#     Ref: K. Fukami and K. Taira,
#     “Grasping extreme aerodynamics on a low-dimensional manifold,”
#     in review, 2023
#
# The code is written for educational clarity and not for speed.
# -- version 1: Aug 11, 2023

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.layers import Input, Add, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Reshape, LSTM, Concatenate, Conv2DTranspose
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm as tqdm
from scipy.io import loadmat


#import tensorflow._api.v2.compat.v1 as tf

#tf.disable_v2_behavior()


### CNN-MLP autoencoder with observable augmentation

act = 'tanh'
input_img = Input(shape=(120,240,1)) # defines the shape of the input tensor for your neural network model. 1 is the number of channels. If you want to swork with u and v, you should use 2 there.

x1 = Conv2D(32, (3,3),activation=act, padding='same')(input_img) # This is a Keras layer for 2D convolution. It applies a specified number of filters (in this case, 32) to the input tensor using a 2D convolutional kernel.
# (32, (3,3)): specifies the number of filters and the size of the convolutional kernel. Here, 32 indicates that the layer will have 32 filters, and (3,3) specifies that each filter will have a size of 3x3.
# Chat gpt says that the dimension of the x1 is (120,240,32)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1) # Max-pooling is a downsampling operation that reduces the spatial dimensions of the input tensor by taking the maximum value within a window.
# (2, 2): This parameter specifies the size of the pooling window. In this case, it's a 2x2 window, meaning that the max-pooling operation will be applied to non-overlapping 2x2 regions of the input tensor.
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)

x1 = MaxPooling2D((2,2),padding='same')(x1)

x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((5,5),padding='same')(x1)
# After this the dimension is (6,12,8)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)

x1 = Reshape([12*6*4])(x1)
x1 = Dense(256,activation=act)(x1)
x1 = Dense(128,activation=act)(x1)
x1 = Dense(64,activation=act)(x1)
x1 = Dense(32,activation=act)(x1)

x_lat = Dense(3,activation=act)(x1)

x_CL = Dense(32,activation=act)(x_lat)
x_CL = Dense(64,activation=act)(x_CL)
x_CL = Dense(32,activation=act)(x_CL)
x_CL_final = Dense(1)(x_CL)
# This is another output that we want the NN to have, as a function of latent variable

x1 = Dense(32,activation=act)(x_lat)
x1 = Dense(64,activation=act)(x1)
x1 = Dense(128,activation=act)(x1)
x1 = Dense(256,activation=act)(x1)
x1 = Dense(288,activation=act)(x1)
x1 = Reshape([6,12,4])(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((5,5))(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x_final = Conv2D(1, (3,3),padding='same')(x1)

autoencoder = Model(input_img, [x_final,x_CL_final])

# The compile method in Keras is used to configure the learning process of a model. 
autoencoder.compile(optimizer='adam', loss='mse',loss_weights=[1,0.05]) # beta = 0.05 determined by L-curve analysis
# loss_weights: [1, 0.05] is a list specifying the weighting for each of the losses provided in the model. In this case, there are two losses: one for the image reconstruction (x_final) and one for the classification (x_CL_final). The weight for the classification loss is set to 0.05, meaning it contributes less to the total loss compared to the reconstruction loss. This kind of setup is useful when you want to balance multiple objectives during training.

num_snap = ABC; ## number of training snapshots

y_1 = np.zeros((num_snap,120,240,1)) # vorticity field
y_CL = np.zeros((num_snap,1)) # lift response


from keras.callbacks import ModelCheckpoint,EarlyStopping
X_train, X_test, X_train1, X_test1 = train_test_split(y_1, y_CL, test_size=0.2, random_state=None) # Note here that ßßß
# X_train and X_test is for time snapshots and X_train1 and X_test1 is the lift parameter


model_cb=ModelCheckpoint('./Model.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
# ModelCheckpoint: This is the callback class responsible for saving the model's weights during training.
# './Model.hdf5': This specifies the path where the model weights will be saved. In this case, it's saving the weights to a file named "Model.hdf5" in the current directory.
# monitor='val_loss': This specifies the quantity to monitor during training. In this case, it's monitoring the validation loss. The callback will save the model weights whenever the validation loss improves.
# save_best_only=True: This indicates that only the best model (according to the monitored quantity) will be saved. If set to True, it will overwrite the previous best model saved.
# verbose=1: This controls the verbosity of the output during training. Setting it to 1 means that it will print messages when a model is saved.

early_cb=EarlyStopping(monitor='val_loss', patience=200,verbose=1)
# EarlyStopping: This is the callback class responsible for stopping the training process when a monitored quantity has stopped improving.
# monitor='val_loss': This specifies the quantity to monitor during training. In this case, it's monitoring the validation loss.
# patience=200: This parameter determines the number of epochs with no improvement after which training will be stopped. In this case, it's set to 200 epochs. So, if the validation loss does not improve for 200 consecutive epochs, training will stop.
# verbose=1: This controls the verbosity of the output during training. Setting it to 1 means that it will print messages when training stops due to early stopping.
cb = [model_cb, early_cb]
history = autoencoder.fit(X_train,[X_train,X_train1],epochs=50000,batch_size=128,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_test, [X_test,X_test1]))
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./History.csv',index=False)


