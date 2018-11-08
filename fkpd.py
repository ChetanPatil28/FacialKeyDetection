import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split

df=pd.read_csv('training.csv')
df['Image']=df['Image'].apply(lambda i:np.fromstring(i,sep=" "))
df=df.dropna()
X=np.vstack(df['Image'].values)/255
X=X.reshape(X.shape[0],96,96)
X=X.astype('float32')
Yy=df.iloc[:,:-1].values
Y = (Yy - 48) / 48 

def rotate(X,Y):
    newX = X
    newY = Y
    for i in range(X.shape[0]):
        newY[i] =  newY[i]*48 + 48
        rotation = np.random.randint(-5,5)
        Mat = cv2.getRotationMatrix2D((48,48), rotation, 1.0)
        newX[i] = cv2.warpAffine(newX[i],Mat,(96,96))
        for j in range(15):
            coord_idx = 2*j
            old_coord =  newY[i][coord_idx:coord_idx+2]
            new_coord = np.matmul(Mat,np.append(old_coord,1))
            newY[i][coord_idx] = new_coord[0]
            newY[i][coord_idx+1] = new_coord[1]
        newY[i] = ( newY[i] - 48)/48
    return  newX,  newY


def horizontal_flip(data, labels):
    newdata = data[:,:,::-1]
    newl = np.zeros(labels.shape)
    for i in range(data.shape[0]):
        newl[i] += labels[i]
        newl[i, 0::2] *= -1
        flip_indices = [(0, 2), (1, 3),(4, 8), (5, 9), (6, 10), (7, 11),(12, 16), (13, 17), 
                        (14, 18), (15, 19),(22, 24), (23, 25),]
        for a,b in flip_indices:
            newl[i,a], newl[i,b] = newl[i,b], newl[i,a]
    return newdata, newl

horx,hory=horizontal_flip(X, Y)
X=np.concatenate((X,horx),axis=0)
Y=np.concatenate((Y,hory),axis=0)
rotX,rotY=rotate(X,Y)
X=np.concatenate((X,rotX),axis=0)
Y=np.concatenate((Y,rotY),axis=0)

X=X.reshape(X.shape[0],96,96,1)

Xtrain,Xval,Ytrain,Yval=train_test_split(X,Y,random_state=7)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(30, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(30))
model.compile('adam', loss='mean_squared_error')
model.fit(X,Y,batch_size=128,epochs=25)
model.save('fkpd.h5')


test=pd.read_csv('test.csv')
test['Image']=test['Image'].apply(lambda i:np.fromstring(i,sep=" "))
test=test.dropna()
Xte=np.vstack(test['Image'].values)/255
Xte=Xte.reshape(Xte.shape[0],96,96,1)
Xte=Xte.astype('float32')

#if input is a set of images, then
preds=model.predict(Xte)[0]
preds=np.squeeze(preds*48+48)

#if input is just an image where 'i' is an index of the image 
pred=model.predict(np.expand_dims(Xte[i]))[0]
pred=np.squeeze(pred*48+48)



