from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

import dogsvcats

IMGSIZE = (150, 150)

# Load the data and make sure its the right size
# 16 GB RAM, no other programs running, can take full dataset, otherwise scale appropriately
Xtr, ytr = dogsvcats.load_train_opt('../input/train/', IMGSIZE, grayscale=False, shuffle=True, load=.2)
assert(Xtr.shape[1:] == IMGSIZE + (3,))

# Split into train and validation (Memory Efficient)
split_ind = int(Xtr.shape[0] * .8)
Xval, yval = Xtr[split_ind:], ytr[split_ind:]
Xtr, ytr = Xtr[:split_ind], ytr[:split_ind]
# print the stats of the data
print("Training Data")
dogsvcats.print_stats(Xtr, ytr, vis=False)
print("Validation Data")
dogsvcats.print_stats(Xval, yval, vis=False)

# Split into train and validation
# NOT MEMORY EFFICIENT
# Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr)

# Instantiate the image augmentor with the desired settings
print("Instantiating the image augmentor")
datagen = ImageDataGenerator(
        rotation_range=20.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Create the training image generator
print("Flowing the augmentor from data")
train_generator = datagen.flow(Xtr, ytr)

# Create the CNN
print("Creating the model")
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# This will save the best scoring model weights to the parent directory
best_model_file = '../CatDogCNN_weights.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True)


# Train the model
print("Training the model")
model.fit_generator(train_generator,
                    samples_per_epoch=Xtr.shape[0],
                    nb_epoch=25,
                    validation_data=(Xval, yval),
                    callbacks=[best_model])

Xtr, Xval, ytr, yval = [None]*4

# Load the test dataset and flatten the images
Xte = dogsvcats.load_test(grayscale=True)


# Load the best scoring model for creating a submission
model.load_weights(best_model_file)

# Make predictions and create a submission
predictions = model.predict_proba(Xte)
dogsvcats.create_submission(predictions)

