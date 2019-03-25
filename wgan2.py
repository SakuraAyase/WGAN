from keras.datasets import *
from keras.models import *
import skimage.io as io
import skimage.data as dt
import skimage.transform as tra
from keras.engine import *
from keras import *
import numpy as np
from scipy import *
from scipy.misc import *

from keras.backend import *
from keras.layers import *
import tensorflow as tf
from PIL import Image

from keras.optimizers import *

from tensorflow.examples.tutorials.mnist import input_data


def d_loss(y_true,y_pred):
    return 2*mean((0.5 - y_true)*y_pred)

def g_loss(y_true,y_pred):
    return -mean(y_pred)

def plot(image,i):
    fileName = 'C:\\Users\\myfamily\\Desktop\\新建文件夹2\\'
    io.imsave(fileName+'lean'+str(i)+'.jpg',image)


y_train = input_data.read_data_sets("mnist").train.labels
x_train = input_data.read_data_sets("mnist",\
        	one_hot=True).train.images
x_train = x_train.reshape(-1, 28,\
        	28, 1).astype(np.float32)

img1 = imread("E:\文件\往届肝\往届肝\ZACA娘\往届图\\test (1).jpg")
x_train = [img1]
for i in range(2,33):
    img = imread("E:\文件\往届肝\往届肝\ZACA娘\往届图\\test ("+str(i)+").jpg")
    print(img.shape)
    x_train.append(img)
print(len(x_train))

for i in range(len(x_train)):
    x_train[i] = x_train[i]/255

x_train = array(x_train)
print(x_train[1].shape)
print(x_train[1])


D = Sequential()
dropout = 0.5
col = 100
row = 100
D.add(Conv2D(128,kernel_size=(2,2),strides=2,padding='same',input_shape=(row,col,3),activation='relu'))

D.add(Conv2D(128,kernel_size=(2,2),strides=1,padding='same',activation='relu'))

D.add(Flatten())
D.add(Dense(1))
D.summary()

G = Sequential()

G.add(Dense(row*col*16,input_dim=100,activation='relu'))
G.add(Reshape((int(row/4),int(col/4),256)))



G.add(UpSampling2D())
G.add(Conv2DTranspose(128,(2,2),padding='same',activation='relu'))


G.add(UpSampling2D())
G.add(Conv2DTranspose(64,(2,2),padding='same',activation='relu'))



G.add(Conv2DTranspose(3,(2,2),padding='same',activation='sigmoid'))

G.summary()

DM = Sequential()
DM.add(D)
DM.compile(loss='mse',optimizer='RMSprop',metrics=['accuracy'])

AM = Sequential()
AM.add(G)
AM.add(D)
AM.compile(loss='mse',optimizer='RMSprop',metrics=['accuracy'])

GM = Sequential()
GM.add(G)
GM.compile(loss='mse',optimizer='RMSprop',metrics=['accuracy'])

batch = 2
size = 32
images_train = x_train[np.random.randint(0,x_train.shape[0], size=size), :, :,:]
noise = np.random.uniform(-1.0, 1.0, size=[size, 100])
GM.fit(noise,images_train,epochs=20,batch_size=batch)
k = GM.predict(noise)
plot(k[0],0)

n_cli = 3
for i in range(2000):
    noise = np.random.uniform(-1.0, 1.0, size=[size, 100])
    img_faker = G.predict(noise)

    images_train = x_train[np.random.randint(0,x_train.shape[0], size=size), :, :, :]
    x = np.concatenate((images_train,img_faker))
    y = np.zeros((size*2,1))
    y[:size,:] = 1
    print('DM')
    for n in range(n_cli):
        DM.fit(x, y, epochs=1, batch_size=batch)
        for I in D.layers:
            weight = I.get_weights()
            weight = [np.clip(w, -0.5,0.5) for w in weight]
            I.set_weights(weight)

    print(DM.predict(x)[0])
    print(DM.predict(x)[size-1])
    print(img_faker[1])
    print(images_train[1])

    y = np.ones((size,1))

    print('AM')
    AM.fit(noise,y,epochs=1,batch_size=batch)
    noise = np.random.uniform(-1.0, 1.0, size=[1, 100])
    k = G.predict(noise)[0]

    plot(k,i+1)
    print(i)



