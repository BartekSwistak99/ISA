
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from glob import glob
from pathlib import Path
from skimage import io
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dropout, Input, MaxPooling2D, Flatten,
                                     UpSampling2D, concatenate, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, Accuracy, Precision, Recall, MeanIoU
from tensorflow.keras.activations import sigmoid, softmax, tanh
import threading
import time


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session =tf.compat.v1.InteractiveSession(config=config)

"""
    img_0 - 0 == brak znaku
    img_1 - 1 == ograniczenie do 20 
    img_2 - 2 == ograniczenie do 30 
    img_3 - 3 == przejście dla pieszych
    img_4 - 4 == rondo
    img_5 - 5 == stop
"""
SIGNS = {
    0: 'brak',
    1: 'ograniczenie do 20',
    2: 'ograniczenie do 30',
    3: 'przejście dla pieszych',
    4: 'rondo',
    5: 'stop'
}

def load_dataset(path, num_of_clasess = 6):
    x, y = [], []
    for im_name in glob(str(Path(path) / '*.png')):
        im = io.imread(im_name) / 255.0
        label = np.zeros(num_of_clasess)
        label[int(im_name[-5])] = 1.0
        x.append(im)
        y.append(label)
    x, y = np.array(x), np.array(y)
    return x, y

def accuracy_sklearn(y_pred, y_true):
    y_true = [np.argmax(y_true[i]) for i in range(y_true.shape[0])]
    y_pred = [np.argmax(y_pred[i]) for i in range(y_pred.shape[0])]
    return accuracy_score(y_pred, y_true)


def translate_network_to_sign_to_word(prediction: np.ndarray):
    return SIGNS[prediction.argmax()]

# In: (128, 128, 3) -> Out: (6,)
def network_model(input_shape = (128, 128, 3), num_of_clasess = 6, loss='categorical_crossentropy', optimizer='adam'):
    inputs = Input(shape=input_shape)

    x = Conv2D(8, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    outputs = Dense(num_of_clasess, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    model.summary()

    return model

def train_sign_recognition(path, model_path_save='model_sign.h5', num_of_clasess = 6, epochs=30):
    x, y = load_dataset(path, num_of_clasess)
    model = network_model(num_of_clasess=num_of_clasess)
    history = model.fit(x, y, batch_size=16, epochs=epochs)
    model.save(model_path_save)
    print('Model saved', model_path_save)

def load_model(path):
    return tf.keras.models.load_model(path)



def cnn_data_consumer(cv):
    with cv:
        cv.wait()
        print("Consumed")

def cnn_data_producer(cv):
    with cv:
        print("Producer produced")
        cv.notifyAll()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    
    #train_sign_recognition('F:\Deep Learning\Data\webots_sign')
    

    