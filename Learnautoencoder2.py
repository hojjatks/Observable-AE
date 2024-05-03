#%%
# autoencoder using only Dense NN
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from ProcessFunctions import ReadData,get_V_theta_tau_t_Nx_Nz
from sklearn.model_selection import train_test_split




#%% importing pickle data
v=np.load("v_T100filtered.npy")
v=np.log10(v[:,:,128-16:128+16])
v_bar=np.mean(v,axis=0)
v=v-v_bar
#%%







# %%
# This is the size of our encoder representation
encoding_dim = 10  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# This is our input image
input_img = keras.Input(shape=(1024,))
encoded = layers.Dense(encoding_dim, activation='tanh')(input_img)
decoded = layers.Dense(1024, activation='tanh')(encoded)
autoencoder = keras.Model(input_img, decoded)
# This model maps an input to its encoded representation
#%% Making the encoder model
encoder = keras.Model(input_img, encoded)
# This is our encoded (32-dimensional) input
#%% Making the decoder model
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
# %%
autoencoder.compile(optimizer='adam', loss='mse')
#%% Loading data and changing its type to float
#(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_test = train_test_split(v, test_size=0.2, random_state=None) # Note here that ßßß

#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
# %%
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
# %%
# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
#%%
# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('./Figs/reconstruct.png')
# %%
