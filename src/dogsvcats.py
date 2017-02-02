import os
from PIL import Image

import numpy as np
from keras.preprocessing.image import img_to_array


def read_image(file_path, img_size=None, grayscale=False):
    # Do some typechecking
    if img_size is not None and ((not isinstance(img_size, tuple)) or len(img_size) != 2):
        raise TypeError("img_size must be a tuple of (w, h); given %s" % img_size)
    # Open the image and convert it to the requested form and size
    img = Image.open(file_path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    if img_size is not None:
        img = img.resize(img_size, Image.BICUBIC)

    return img


def prep_data(images, img_size=(64, 64), grayscale=False):
    count = len(images)
    if grayscale:
        channels = 1
    else:
        channels = 3
    data = np.empty((count, img_size[1], img_size[0], channels), dtype=np.uint8)
    for i, image_file in enumerate(images):
        image = read_image(image_file, img_size=img_size, grayscale=grayscale)
        data[i] = img_to_array(image)
        if i % 1000 == 0: print('Processed {} of {}'.format(i, count))

    return data


def load_train(train_dir='../input/train/', img_size=(64, 64), grayscale=False, shuffle=True):
    X_list = []
    y_list = []
    for i, class_name in enumerate(('cat', 'dog')):
        img_fpaths = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if class_name in fname]
        X_list.append(prep_data(img_fpaths, img_size=img_size, grayscale=grayscale))
        y_list.append(np.full((len(img_fpaths),), float(i), dtype=float))
    X = np.vstack(X_list) / 255.
    y = np.hstack(y_list)
    if shuffle:
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        X = X[inds]
        y = y[inds]
    return X, y


def load_test(test_dir='../input/test/', img_size=(64, 64), grayscale=False):
    img_fpaths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir)]
    return prep_data(img_fpaths, img_size=img_size, grayscale=grayscale) / 255.


if __name__ == "__main__":
    IMG_SIZE = (32, 32)
    for g in (True, False):
        Xtr, ytr = load_train(img_size=IMG_SIZE, grayscale=g)
        print("Train shape: {}".format(Xtr.shape))
        print("Label Shape: {}".format(ytr.shape))
        print("Max Value: {}".format(np.max(Xtr)))
