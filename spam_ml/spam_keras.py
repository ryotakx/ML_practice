import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras import losses,optimizers
import gc

def getdata():
    td = open('spam_data\spam_train.csv', "r")
    xtrain,ytrain = [],[]
    for line in td:
        row = line.rstrip('\r\n').split(',')
        for i in range(1, len(row)):
            row[i] = float(row[i])
        mail = row[1:-1]
        label = row[-1]
        xtrain.append(np.array(mail))
        ytrain.append(label)
    return xtrain,ytrain

x,y = getdata()
#x = list(sequence.pad_sequences(x, maxlen=57))
#y = list(sequence.pad_sequences(y, maxlen=1))
x_train = np.array(list(x))[:3000]
y_train = np.array(list(y))[:3000]
x_test = np.array(list(x))[3000:]
y_test = np.array(list(y))[3000:]

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#print(type(x_train[1]))
#print(x_train[1])
model = Sequential()
model.add(Dense(57,activation='sigmoid',input_dim=57))

model.add(Dense(2,activation='softmax'))


model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=128,epochs=40,verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
gc.collect()
print('Test loss:', score[0])
print('Test accuracy:', score[1])