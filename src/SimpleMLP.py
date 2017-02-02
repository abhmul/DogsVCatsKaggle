from keras.layers import Dense
from keras.models import Sequential

import dogsvcats

Xtr, ytr = dogsvcats.load_train('../input/train/', (64, 64), grayscale=True)
# Flattens the input vectors
# TODO Comment the line below if passing to a convnet
Xtr = Xtr.reshape(Xtr.shape[0], -1)

model = Sequential()
### TODO Replace this code here with a ConvNet architecture ###
model.add(Dense(200, activation='relu', input_dim=Xtr.shape[1]))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
### TODO Replace the above code here with a ConvNet architecture ###

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Mess around with the number of epochs to make training better
fit = model.fit(Xtr, ytr, nb_epoch=10, validation_split=.2)

# TODO (Abhijeet) add an early stopping callback
# TODO (Abhijeet) add a create_submission function