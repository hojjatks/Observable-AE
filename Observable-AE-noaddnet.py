
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#%%
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
import pyqdyn
from ProcessFunctions import ReadData,get_V_theta_tau_t_Nx_Nz
from keras.callbacks import ModelCheckpoint,EarlyStopping
import pandas as pd


#%% importing pickle data
data_directory='/groups/astuart/hkaveh/QDYN_autoencoder/dc_0.045.pickle'
p=ReadData(data_directory)
T_filter=100 # remove the first T_filter years
v,theta,tau,t,Nx,Nz=get_V_theta_tau_t_Nx_Nz(p,T_filter)
print("Data is loaded!   :)")


#%%
#import tensorflow._api.v2.compat.v1 as tf

#tf.disable_v2_behavior()


### CNN-MLP autoencoder with observable augmentation

act = 'tanh'
input_img = Input(shape=(Nz,Nx,1)) # defines the shape of the input tensor for your neural network model. 1 is the number of channels. If you want to swork with u and v, you should use 2 there.


Maxpool1=np.array([2,2,4],dtype=int) # Along the depth of the fault
Maxpool2=np.array([2,2,2],dtype=int) # Along the strike of the fault

x1 = Conv2D(32, (3,3),activation=act, padding='same')(input_img) # This is a Keras layer for 2D convolution. It applies a specified number of filters (in this case, 32) to the input tensor using a 2D convolutional kernel.
# (32, (3,3)): specifies the number of filters and the size of the convolutional kernel. Here, 32 indicates that the layer will have 32 filters, and (3,3) specifies that each filter will have a size of 3x3.
# Chat gpt says that the dimension of the x1 is (120,240,32)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((Maxpool1[0],Maxpool2[0]),padding='same')(x1) # Max-pooling is a downsampling operation that reduces the spatial dimensions of the input tensor by taking the maximum value within a window.
# (2, 2): This parameter specifies the size of the pooling window. In this case, it's a 2x2 window, meaning that the max-pooling operation will be applied to non-overlapping 2x2 regions of the input tensor.
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)

x1 = MaxPooling2D((Maxpool1[1],Maxpool2[1]),padding='same')(x1)

x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((Maxpool1[2],Maxpool2[2]),padding='same')(x1)
# After this the dimension is (6,12,8)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)

x1 = Reshape([int(Nz*Nx//(np.prod(Maxpool1)*np.prod(Maxpool2)))*4])(x1)

x1 = Dense(256,activation=act)(x1)
x1 = Dense(128,activation=act)(x1)
x1 = Dense(64,activation=act)(x1)
x1 = Dense(32,activation=act)(x1)

x_lat = Dense(3,activation=act)(x1)

# x_CL = Dense(32,activation=act)(x_lat)
# x_CL = Dense(64,activation=act)(x_CL)
# x_CL = Dense(32,activation=act)(x_CL)
# x_CL_final = Dense(1)(x_CL)
# This is another output that we want the NN to have, as a function of latent variable
# I commented it but I might need it for future use to learn time.

x1 = Dense(32,activation=act)(x_lat)
x1 = Dense(64,activation=act)(x1)
x1 = Dense(128,activation=act)(x1)
x1 = Dense(256,activation=act)(x1)
x1 = Dense(int(Nz*Nx//(np.prod(Maxpool1)*np.prod(Maxpool2)))*4,activation=act)(x1)




x1 = Reshape([Nz//np.prod(Maxpool1),int(Nx//np.prod(Maxpool2)),4])(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((Maxpool1[2],Maxpool2[2]))(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((Maxpool1[1],Maxpool2[1]))(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((Maxpool1[0],Maxpool2[0]))(x1)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x1 = Conv2D(32, (3,3),activation=act, padding='same')(x1)
x_final = Conv2D(1, (3,3),padding='same')(x1)

autoencoder = Model(input_img, x_final)
#%%
# The compile method in Keras is used to configure the learning process of a model. 
autoencoder.compile(optimizer='adam', loss='mse',loss_weights=[1]) # beta = 0.05 determined by L-curve analysis
# loss_weights: [1, 0.05] is a list specifying the weighting for each of the losses provided in the model. In this case, there are two losses: one for the image reconstruction (x_final) and one for the classification (x_CL_final). The weight for the classification loss is set to 0.05, meaning it contributes less to the total loss compared to the reconstruction loss. This kind of setup is useful when you want to balance multiple objectives during training.

Ntotsnap = v.shape[0]  ## number of total snapshots

num_val=int(Ntotsnap*0.1)
v_fit=v[:-num_val,:,:] # training and test data
num_snap = v_fit.shape[0]; ## number of training snapshots (training+test set)
y_1 = v_fit.reshape((num_snap,Nz,Nx,1)) # velocity field
# y_CL = np.zeros((num_snap,1)) # lift response
X_train, X_test = train_test_split(y_1, test_size=0.2, random_state=None) # Note here that ßßß
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
# verbose=1: This controls the verbosity of
#  the output during training. Setting it to 1 means that it will print messages when training stops due to early stopping.
cb = [model_cb, early_cb]
#%%

history = autoencoder.fit(X_train,X_train,epochs=1400,batch_size=64,verbose=1,callbacks=cb,shuffle=True,validation_data=(X_test, X_test))
#%%
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./History.csv',index=False)


