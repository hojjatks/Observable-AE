#%%
# This is autoencoder with a small portion of data (cut along strike)
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from ProcessFunctions import ReadData,get_V_theta_tau_t_Nx_Nz
from sklearn.model_selection import train_test_split


#%% 

#%% importing pickle data
v=np.load("v_T100filtered.npy")
v=np.log10(v[:,:,128-16:128+16])
v_bar=np.mean(v,axis=0)
v=(v-v_bar)
#%%





nencode=1
ncolv=3

# error with ncolv=6 in first epoch= 0.673
# error with ncolv=3 in the first iteration is 0.71
# %%
# This is the size of our encoder representation
# = 10  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# This is our input image

input_img = keras.Input(shape=(32, 32, 1))

x = layers.Conv2D(16, (ncolv, ncolv), activation='tanh', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (ncolv, ncolv), activation='tanh', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (ncolv, ncolv), activation='tanh', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Reshape([int(4*4*8)])(x)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
encoded=layers.Dense(nencode,activation="tanh")(x)
x=layers.Dense(int(4*4*8),activation="tanh")(encoded)
x=layers.Reshape([4,4,8])(x)
x = layers.Conv2D(8, (ncolv, ncolv), activation='tanh', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (ncolv, ncolv), activation='tanh', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (ncolv, ncolv), activation='tanh',padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (ncolv, ncolv), activation='tanh', padding='same')(x)
autoencoder = keras.Model(input_img, decoded)
# This model maps an input to its encoded representation
autoencoder.compile(optimizer='adam', loss='mse')

#%% Loading data and changing its type to float
#(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_test = train_test_split(v, test_size=0.2, random_state=None) # Note here that ßßß
x_train = np.reshape(x_train, (len(x_train), 32, 32, 1))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 1))
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#%%
from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
# %%
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#plt.savefig('./Figs/convol.png')
# %%

n_lat=np.array([1])
error=np.array([])
# %%
