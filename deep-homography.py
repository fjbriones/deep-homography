import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, Activation, Lambda
from keras.optimizers import SGD
from keras import backend as K
import glob
import numpy as np
import os.path


# Data loader for training and validation
def data_loader(path):
    while True:
        for npz in glob.glob(os.path.join(path, '*.npz')):
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']/255  # Normalize to 0 <= images <= 1
            offsets = archive['offsets']/32  # Normalize to -1 <= offsets <= 1
            yield images, offsets


# Common architecture to both networks
def common(model):
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    return model


# Last layer of the regression network
def regression_network_last(model, lr):
    model_r = common(model)
    model_r.add(Dense(8, activation='relu'))
    sgd = SGD(lr=lr, momentum=0.9)
    model_r.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])
    return model_r


# For the Lambda function of finding the maximum along the 21x21 distribution
def max_function(x):
    y = K.max(x, axis=2)
    return y


# Last layer of classification network
def classification_network_last(model, lr, batch_size=64):
    model_c = common(model)
    model_c.add(Dense(168))
    model_c.add(Reshape((8, 21)))
    model_c.add(Activation(activation=K.softmax))
    model_c.add(Lambda(max_function))
    sgd = SGD(lr=lr, momentum=0.9)
    model_c.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mse'])
    return model_c


# Model training for regression network
def regression_network(train_data_path, val_data_path, total_iterations, batch_size, itr, steps_per_epoch, val_steps):
    model_r = Sequential()
    model_r = regression_network_last(model_r, lr=0.005)
    for i in range(0, total_iterations - itr, itr):
        div = (10 ** (int(i / itr)))
        model_r.optimizer.lr.set_value = 0.005/div
        epochs = int(itr / steps_per_epoch) + 1
        model_r.fit_generator(generator=data_loader(train_data_path),
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              verbose=1,
                              validation_data=data_loader(val_data_path),
                              validation_steps=val_steps)
        model_r.save('models/regression.h5')


# Model training for classification network
def classification_network(train_data_path, val_data_path, total_iterations, batch_size, itr, steps_per_epoch, val_steps):
    model_c = Sequential()
    model_c = classification_network_last(model_c, lr=0.005)
    for i in range(0, total_iterations - itr, itr):
        div = (10 ** (int(i / itr)))
        model_c.optimizer.lr.set_value = 0.005 / div
        epochs = int(itr / steps_per_epoch) + 1
        model_c.fit_generator(generator=data_loader(train_data_path),
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              verbose=1,
                              validation_data=data_loader(val_data_path),
                              validation_steps=val_steps)
        model_c.save('models/classification.h5')


def test_model(model_save_path, test_data_path, test_size=5000, batch_size=64):
    i = 0
    j = 0
    error = np.empty(int(test_size/batch_size)+1)
    images = np.empty((batch_size, 128, 128, 2))
    offsets = np.empty((batch_size, 8))
    model_l = load_model(model_save_path)
    for npz_test in glob.glob(os.path.join(test_data_path, '*.npz')):
        archive = np.load(npz_test)
        images[i] = np.resize(archive['images'], (128, 128, 2))/255
        offsets[i] = archive['offsets']
        i = i + 1
        if i % batch_size == 0:
            offsets_predicted = model_l.predict(images)*64
            x_1 = np.sqrt((offsets[:, 0] - offsets_predicted[:, 0])**2 + (offsets[:, 1] - offsets_predicted[:, 1])**2)
            x_2 = np.sqrt((offsets[:, 2] - offsets_predicted[:, 2])**2 + (offsets[:, 3] - offsets_predicted[:, 3])**2)
            x_3 = np.sqrt((offsets[:, 4] - offsets_predicted[:, 4])**2 + (offsets[:, 5] - offsets_predicted[:, 5])**2)
            x_4 = np.sqrt((offsets[:, 6] - offsets_predicted[:, 6])**2 + (offsets[:, 7] - offsets_predicted[:, 7])**2)
            error[j] = np.average(x_1 + x_2 + x_3 + x_4)/4
            print('Mean Corner Error: ', error[j])
            j = j + 1
            i = 0
    print('Mean Average Corner Error: ', np.average(error))


train_data_path = 'train-data-combined/'
val_data_path = 'val-data-combined/'
test_data_path = 'test-data/'

total_iterations = 90000
batch_size = 64

train_samples = 500000
steps_per_epoch = int(train_samples / batch_size)

val_samples = 50000
val_steps = int(val_samples / batch_size)

itr = 30000

# print("Starting Training of Regression Network...")
# regression_network(train_data_path, val_data_path, total_iterations, batch_size, itr, steps_per_epoch, val_steps)

# print("Starting Training of Classification Network...")
# classification_network(train_data_path, val_data_path, total_iterations, batch_size, itr, steps_per_epoch, val_steps)

print("Testing the Regression Network...")
test_model(model_save_path='models/regression.h5', test_data_path='test-data/')

print("Testing the Classification Network...")
test_model(model_save_path='models/classification.h5', test_data_path='test-data/')
