#%%
from keras.layers import Input, Add, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Reshape, LSTM, Concatenate, Conv2DTranspose
from keras.models import Model
#from keras import backend as K
import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
import tensorflow as tf
#from tqdm import tqdm as tqdm
from scipy.io import loadmat
import pyqdyn
from ProcessFunctions import ReadData,get_V_theta_tau_t_Nx_Nz
from keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt
#%%
model = tf.keras.models.load_model('./Model_Test4.hdf5')

# %% Loading testing data
#%% importing pickle data
data_directory='/groups/astuart/hkaveh/QDYN_autoencoder/dc_0.045.pickle'
p=ReadData(data_directory)
T_filter=100 # remove the first T_filter years
v,_,_,_,Nx,Nz=get_V_theta_tau_t_Nx_Nz(p,T_filter)
#v=np.log(v) # We work on the log of the data
print("Data is loaded!   :)")
Ntotsnap = v.shape[0]  ## number of total snapshots
num_val=int(Ntotsnap*0.1)
#v_fit=v[:-num_val,:,:] # training and test data
v_test=v[-num_val:,:,:] # Test set, the one that is not used in the ML
#%% Check the result:

plt.figure(figsize=(12, 6))
plt.title("Model_Test4")
x = p.ox["x"].unique()
z = p.ox["z"].unique()
X, Z = np.meshgrid(x, z) 
test=v_test[1000]
out_test=model.predict([test.reshape(-1,Nz,Nx,1)])[0,:,:,0]


plt.subplot(2, 1, 1)  # Create the first subplot in a 1x2 grid (1 row, 2 columns), index 1
contour_plot = plt.contourf(X,Z,test, cmap='coolwarm')
plt.subplot(2, 1, 2)  # Create the first subplot in a 1x2 grid (1 row, 2 columns), index 1
contour_plot = plt.contourf(X,Z,out_test, cmap='coolwarm')

plt.show()

#%%

encoder = Model(inputs=model.input, outputs=model.get_layer('dense_4').output)
zeta=encoder.predict(v_test.reshape(v_test.shape[0],Nz,Nx,1))

#%%
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')
ax.plot(zeta[:,0], zeta[:,1], zeta[:,2], c='b', linewidth=1)
# ax.scatter(x_lat_test[:,0], x_lat_test[:,1], x_lat_test[:,2], c='r', marker='o')
# ax.set_xlim(-2, 2)
# ax.set_ylim(0.9, 1)
# ax.set_zlim(-2, 2)
ax.set_xlabel(r"$\xi_1$")
ax.set_ylabel(r"$\xi_2$")
ax.set_zlabel(r"$\xi_3$")
plt.tight_layout()
#plt.subplots_adjust(left=0.1, right=2.9, top=0.9, bottom=0.1)
#plt.savefig('latent_variables.pdf')
plt.show()
#%%
#loss_history = pd.read_csv('History_Simplified.csv', delimiter=',')
#plt.figure(figsize=(8, 4))
#plt.plot(loss_history["epoch"],loss_history["loss"],linewidth=6,label='training loss')
#plt.plot(loss_history["epoch"],loss_history["val_loss"],color='red', linewidth=2,label='validation loss', linestyle='--')
#plt.xlabel('Epochs')
#plt.ylabel(r'Error $L^2$ norm')
#plt.legend()
#plt.savefig('loss.pdf')
#plt.show()
# %%
