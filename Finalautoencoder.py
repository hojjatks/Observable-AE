#%%
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from ProcessFunctions import ReadData,get_V_theta_tau_t_Nx_Nz
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint,EarlyStopping
import pandas as pd
import tensorflow as tf

#%% 

#%% importing pickle data
v=np.load("v_T100filtered.npy")
v=np.log10(v)
v_bar=np.mean(v,axis=0)
v=(v-v_bar)
#%%





nencode=5
ncolv=3

# error with ncolv=6 in first epoch= 0.673
# error with ncolv=3 in the first iteration is 0.71
# %%
# This is the size of our encoder representation
# = 10  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# This is our input image

input_img = keras.Input(shape=(32, 256, 1))

x = layers.Conv2D(16, (ncolv, ncolv), activation='tanh', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (ncolv, ncolv), activation='tanh', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (ncolv, ncolv), activation='tanh', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Reshape([int(4*32*8)])(x)
x = layers.Dense(256,activation="tanh")(x)
x = layers.Dense(128,activation="tanh")(x)
x = layers.Dense(64,activation="tanh")(x)
x = layers.Dense(32,activation="tanh")(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
encoded=layers.Dense(nencode,activation="tanh")(x)
x=layers.Dense(32,activation="tanh")(encoded)
x=layers.Dense(64,activation="tanh")(encoded)
x=layers.Dense(128,activation="tanh")(encoded)
x=layers.Dense(256,activation="tanh")(encoded)

x=layers.Dense(int(4*32*8),activation="tanh")(x)


x=layers.Reshape([4,32,8])(x)
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
x_train = np.reshape(x_train, (len(x_train), 32, 256, 1))
x_test = np.reshape(x_test, (len(x_test), 32, 256, 1))
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#%%
from keras.callbacks import TensorBoard
model_cb=ModelCheckpoint('./Model_new.hdf5', monitor='val_loss',save_best_only=True,verbose=1)


history=autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),callbacks=[model_cb])
# %%
decoded_imgs = autoencoder.predict(x_test)

n = 4
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(32, 256))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(32, 256))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('./Figs/convol'+str(nencode)+'.png')
# %%

n_lat=np.array([2,3,4,5,6])
error=np.array([0.71,0.66,0.62,0.60,0.58,0.57])
# %%
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./History_new.csv',index=False)
#%%
loss_history = pd.read_csv('History_new.csv', delimiter=',')
plt.figure(figsize=(8, 4))
plt.plot(loss_history["epoch"],loss_history["loss"],linewidth=6,label='training loss')
plt.plot(loss_history["epoch"],loss_history["val_loss"],color='red', linewidth=2,label='validation loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel(r'Error')
plt.legend()
plt.savefig('Figs/loss_nencode'+str(nencode)+'.png')
plt.show()

# %%
autoencoder = tf.keras.models.load_model('./Model_new.hdf5')
encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_4').output)
x_lat_data_train = encoder.predict(x_train)
x_lat_data_test  = encoder.predict(x_test)
#%%
fig = plt.figure(figsize=(5, 4))

ax = fig.add_subplot(111)
ax.scatter(x_lat_data_train[:1000,0], x_lat_data_train[:1000,1], x_lat_data_train[:1000,2], c='b', linewidth=3)
ax.scatter(x_lat_data_test[:,0], x_lat_data_test[:,1], x_lat_data_test[:,2], c='r', linewidth=3)
# ax.scatter(x_lat_test[:,0], x_lat_test[:,1], x_lat_test[:,2], c='r', marker='o')
# ax.set_xlim(-2, 2)
# ax.set_ylim(0.9, 1)
# ax.set_zlim(-2, 2)
ax.set_xlabel(r"$\xi_1$")
ax.set_ylabel(r"$\xi_2$")
#ax.set_zlabel(r"$\xi_3$")
plt.tight_layout()
#plt.subplots_adjust(left=0.1, right=2.9, top=0.9, bottom=0.1)
plt.show()
plt.savefig('./Figs/latent_variables_nencode'+str(nencode)+'.png')

# %%
