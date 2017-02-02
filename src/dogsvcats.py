import os
from PIL import Image
import datetime

import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array


def read_image(file_path, img_size=None, grayscale=False):
    """
    Reads an image from a filepath with specified settings
    :param file_path: path to the image
    :param img_size: size image should be loaded to
    :param grayscale: Whether or not image should be grayscale
    :return: A PIL Image object
    """
    # Do some typechecking
    if img_size is not None and ((not isinstance(img_size, tuple)) or len(img_size) != 2):
        raise TypeError("img_size must be a tuple of (w, h); given %s" % img_size)
    # Open the image and convert it to the requested form and size
    img = Image.open(file_path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    # Resize if we need to
    if img_size is not None:
        img = img.resize(img_size, Image.BICUBIC)

    return img


def prep_data(images, img_size=(64, 64), grayscale=False):
    """
    Prepares a list of image paths into a 4d numpy array
    :param images: an iterable of image paths
    :param img_size: The size each image should be loaded to
    :param grayscale: Whether or not each image should be grayscale
    :return: a 4d numpy array of shape (num_images, img_height, img_width, num_channels)
    """
    count = len(images)
    # Grayscale has just one channel, RGB has 3
    if grayscale:
        channels = 1
    else:
        channels = 3
    # Initialize the numpy array that will hold all the image data
    data = np.empty((count, img_size[1], img_size[0], channels), dtype=np.uint8)
    # Iterate through each image, turn it into a numpy array, and insert it into the data numpy array
    for i, image_file in enumerate(images):
        image = read_image(image_file, img_size=img_size, grayscale=grayscale)
        data[i] = img_to_array(image)
        if i % 1000 == 0: print('Processed {} of {}'.format(i, count))

    return data


def load_train(train_dir='../input/train/', img_size=(64, 64), grayscale=False, shuffle=True):
    """
    Loads the training dataset
    :param train_dir: the directory the training images are stored
    :param img_size: The size to load the image to
    :param grayscale: Whether or not image should be grayscale
    :param shuffle: Whether or not to shuffle the dataset before returning
    :return: A tuple of the image 4d numpy array and a 1d array of binary labels
    """
    X_list = []
    y_list = []
    # For each class load the image data
    # i = 0 for cat, i = 1 for dog
    for i, class_name in enumerate(('cat', 'dog')):
        # Get a list of filepaths for images of the specific class
        img_fpaths = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if class_name in fname]
        # Add the image data to our list of numpy array data
        X_list.append(prep_data(img_fpaths, img_size=img_size, grayscale=grayscale))
        # Add the labels to our label list
        y_list.append(np.full((len(img_fpaths),), float(i), dtype=float))
    # Merge the dog and cat numpy arrays
    # Normalize the pixel values in the image to be in [0,1]
    X = np.vstack(X_list) / 255.
    y = np.hstack(y_list)
    # Shuffle the data if we need to
    if shuffle:
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds]
        y = y[inds]
    return X, y


def load_test(test_dir='../input/test/', img_size=(64, 64), grayscale=False):
    """
    Loads the test data
    :param test_dir: the directory the test images are stored
    :param img_size: The size to load the image to
    :param grayscale: Whether or not image should be grayscale
    :return: A 4d numpy array of image data
    """
    img_fpaths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir)]
    return prep_data(img_fpaths, img_size=img_size, grayscale=grayscale) / 255.


def create_submission(predictions, info=''):
    """
    Creates a submission for kaggle and saves it in the current directory
    :param predictions: The numpy array of predictions output by the NN
    :param info: An optional parameter to add info to submission filename
    """
    # Turn the predictions into a dataframe
    result1 = pd.DataFrame(predictions, columns=['label'])
    # Kaggle wants the id numbers to start from 1
    result1.index += 1
    # Get the time of submission creation
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    # Save the submission as a csv file of the proper format
    result1.to_csv(sub_file, index=True, index_label='id')

if __name__ == "__main__":
    # Just some informal testing stuff
    IMG_SIZE = (32, 32)
    for g in (True, False):
        Xtr, ytr = load_train(img_size=IMG_SIZE, grayscale=g)
        print("Train shape: {}".format(Xtr.shape))
        print("Label Shape: {}".format(ytr.shape))
        print("Max Value: {}".format(np.max(Xtr)))
