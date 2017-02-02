from keras.layers import Dense
from keras.models import Sequential

import dogsvcats

Xtr, ytr = dogsvcats.load_train('../input/train/', (64, 64), grayscale=True)
# Flattens the input vectors
Xtr = Xtr.reshape(Xtr.shape[0], -1)

model = Sequential()
model.add(Dense(200, activation='relu', input_dim=Xtr.shape[1]))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

fit = model.fit(Xtr, ytr, nb_epoch=10, validation_split=.2)