from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

import dogsvcats

Xtr, ytr = dogsvcats.load_train('../input/train/', (64, 64), grayscale=True)
# Flattens the input vectors
# TODO Comment the line below if passing to a convnet
Xtr = Xtr.reshape(Xtr.shape[0], -1)

model = Sequential()
# TODO Replace this code here with a ConvNet architecture
model.add(Dense(200, activation='relu', input_dim=Xtr.shape[1]))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# This will save the best scoring model weights to the parent directory
best_model_file = '../CatDogNet_weights.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True)

# Mess around with the number of epochs to make training better
fit = model.fit(Xtr, ytr, nb_epoch=10, validation_split=.2, callbacks=[best_model])

# If you run into memory issues, uncommenting the line below might help
# Xtr, ytr = None, None

# Load the test dataset and flatten the images
Xte = dogsvcats.load_test(grayscale=True)
# TODO Comment the line below if passing to a convnet
Xte = Xte.reshape(Xte.shape[0], -1)


# Load the best scoring model for creating a submission
model.load_weights(best_model_file)

# Make predictions and create a submission
predictions = model.predict_proba(Xte)
dogsvcats.create_submission(predictions)
