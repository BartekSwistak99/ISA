import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.color import rgb2hsv, rgb2gray
from scipy.ndimage.measurements import find_objects
from skimage.measure import label
from skimage import io
from glob import glob
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dropout, Input, MaxPooling2D,
                                     Softmax, UpSampling2D, concatenate, Dense, Flatten)


# ----------------------------------------------------------------------------- #
#                            CNNClassifier                                      #
# ----------------------------------------------------------------------------- #
def CNNClassifier(input_shape = (20, 20, 1), num_of_clasess = 5, loss='categorical_crossentropy', optimizer='adam'):
    inputs = Input(shape=input_shape)

    x = Conv2D(8, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    outputs = Dense(num_of_clasess, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    model.summary()

    return model

def accuracy_sklearn(y_pred, y_true):
    y_true = [np.argmax(y_true[i]) for i in range(y_true.shape[0])]
    y_pred = [np.argmax(y_pred[i]) for i in range(y_pred.shape[0])]
    return accuracy_score(y_pred, y_true)


def trainingCNNClassifier(dataset_path:str='dataset_segmented_sign.dat', model_save:str='classifierMod.h5'):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
        x,y = dataset
        x = np.array(x) / 255.0
        y = np.array(y)

        model = CNNClassifier()
        model.fit(x,y, epochs=150)

        y_pred = model.predict(x)
        print(accuracy_sklearn(y_pred, y))


        model.save(model_save)
    exit()

# ----------------------------------------------------------------------------- #
#                                   UNet2D                                      #
# ----------------------------------------------------------------------------- #
def dice_coef(y_true, y_pred, smooth=1e-5):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = 2 * K.sum(K.abs(y_true * y_pred), axis=-1) + smooth
    sums = (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    return intersection / sums

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def UNet2DBlock(inputs, layers, filters, 
    kernel_size=(3, 3), activation='relu', padding='same', 
    kernel_initializer='he_normal', dropout=0.2
):
    conv = inputs
    for _ in range(layers):
        conv = Conv2D(filters, kernel_size,
                      padding=padding, kernel_initializer=kernel_initializer)(conv)
        conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)
    conv = Dropout(dropout)(conv)
    return conv

def UNet2DModel(shape = (128, 64, 1), weights=None):
    inputs = Input(shape)

    # encoder
    x = inputs
    conv_encoder = []
    for filters in [8, 16, 32, 64]:
        conv = UNet2DBlock(x, layers=2, filters=filters, dropout=0.25)
        x = MaxPooling2D(pool_size=(2, 2))(conv)
        conv_encoder.append(conv)

    x = UNet2DBlock(x, layers=2, filters=128, dropout=0.5)

    # decoder
    for filters in [128, 64, 32, 16]:
        x = UNet2DBlock(x, layers=2, filters=filters, dropout=0.25)
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([conv_encoder.pop(), x])

    x = UNet2DBlock(x, layers=1, filters=8, dropout=0.25)
    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=dice_loss, metrics = ['acc'])

    model.summary()
    if weights is not None:
        model.load_weights(weights)
    return model

def training_UNet2D(path_dataset:str = 'dataset_sign_2.dat', model_save:str = 'modelSeg.h5'):
    with open(path_dataset, 'rb') as f:
        dataset = pickle.load(f)

        x, y = dataset

        x = np.array(x)
        y = np.array(y)

        model = UNet2DModel()
        model.fit(x, y, batch_size=16, epochs=50)
        model.save(model_save)


# ----------------------------------------------------------------------------- #
#                                   Other                                       #
# ----------------------------------------------------------------------------- #

def image_to_dataset():
    X = []
    Y = []

    y = None
    x = None
    with open('data_pre_label.dat', 'rb') as f:
        y = pickle.load(f)
    with open('data_pre_image.dat', 'rb') as f:
        x = pickle.load(f)

    for i in range(len(y)):
        seg = rgb2hsv(y[i])
        new = seg[:,:,0] * seg[:,:,1]
        new[np.where(new < 0.85)] = 0
        new[np.where(new > 0.0)] = 1.0
        #new = getLargestCC(new)
        new = new[..., np.newaxis]
        xnew = rgb2gray(x[i]) 
        xnew = xnew[..., np.newaxis]

        new_x = xnew[128:,64:128]
        new_y = new[128:,64:128]

        X.append(new_x)
        Y.append(new_y)
    
    y = None
    x = None
    with open('data_pre_label_2.dat', 'rb') as f:
        y = pickle.load(f)
    with open('data_pre_image_2.dat', 'rb') as f:
        x = pickle.load(f)
    for i in range(len(y)):
        seg = rgb2hsv(y[i])
        new = seg[:,:,0] * seg[:,:,1]
        new[np.where(new < 0.85)] = 0
        new[np.where(new > 0.0)] = 1.0
        new = new[..., np.newaxis]
        xnew = rgb2gray(x[i]) 
        xnew = xnew[..., np.newaxis]

        new_x = xnew[128:,64:128]
        new_y = new[128:,64:128]

        plt.imshow(new_y, 'gray')
        plt.show()

        X.append(new_x)
        Y.append(new_y)

    dataset = [X, Y]

    with open('dataset_sign_2.dat', 'wb') as f:
        pickle.dump(dataset, f)
    exit()


def convert_to_dataset():
    x = []
    y = []

    ldict = {
        's': 0,
        'p': 1,
        '2': 2,
        '3': 3,
        'r': 4
    }
    print(len(ldict))
    for key in ldict:
        for imname in glob(f'labeled/*_{key}.png'):
            im = io.imread(imname, as_gray=True)
            im = im[..., np.newaxis]
            
            y_true = np.zeros(len(ldict))
            y_true[ldict[key]] = 1.0

            print(im.shape, y_true.shape)
            x.append(im)
            y.append(y_true)
    dataset = [x, y]
    with open('dataset_segmented_sign.dat', 'wb') as f:
        pickle.dump(dataset, f)
    
    
    exit()


# ----------------------------------------------------------------------------- #
#                               Prediction                                      #
# ----------------------------------------------------------------------------- #

class ProcessSegmentation():
    def __init__(self, model_path:str):
        self.model:Model = tf.keras.models.load_model(model_path, custom_objects={'dice_coef':dice_coef, 'dice_loss':dice_loss})
        pass

    def getLargestCC(self, segmentation: np.ndarray, min_count = 150) -> np.ndarray:
        """
        min_bicount: 
            minimum object size 
        """
        labels = label(segmentation)
        if labels.max() == 0:
            return np.zeros(segmentation.shape)

        obj_count = np.bincount(labels.flat)[1:] # [1:] to remove background
        obj_count[np.where(obj_count < min_count)] = 0 # remove objects with size less than min_count
        if obj_count.max() == 0:
            return np.zeros(segmentation.shape)

        largestCC = labels == np.argmax(obj_count)+1
        return largestCC

    def getCenterOfSlice(self, slic:slice, max_value:int, box_size=20)-> slice:
        assert (max_value > box_size)
        center = (slic.start + slic.stop) // 2
        center_start = center - box_size // 2
        center_stop = center + box_size // 2
        if center_start < 0:
            return slice(0, box_size, None)
        if center_stop > max_value:
            return slice(max_value - box_size, max_value, None)

        return slice(center_start, center_stop, None)

    def processPrediction(self, prediction:np.ndarray, box_size=20, prob_th=0.5) -> [slice, slice]:
        component = self.getLargestCC(prediction > prob_th)
        if component.max() == 0:
            return None # there isn't any object 
        indexes = find_objects(component)[0]
        x, y, _ = indexes
        x = self.getCenterOfSlice(x, prediction.shape[0], box_size)
        y = self.getCenterOfSlice(y, prediction.shape[1], box_size)
        return x, y

    def getCroppedImageFromPred(self, image:np.ndarray, prediction:np.ndarray, box_size=20, prob_th=0.5) -> np.ndarray:
        bounding_box = self.processPrediction(prediction, box_size, prob_th)
        if bounding_box is None:
            return None
        x,y = bounding_box
        return image[x,y]
    
    def predictAndCrop(self, image:np.ndarray, box_size=20, prob_th=0.5) -> np.ndarray:
        prediction = self.model.predict(np.array([image]))[0]
        cropped_im = self.getCroppedImageFromPred(image, prediction, box_size, prob_th)
        if cropped_im is None:
            return None
        return cropped_im, prediction

    def predictImage(self, images:np.ndarray) -> np.ndarray:
        prediction = self.model.predict(images)
        return prediction

class SignPrediction():
    def __init__(self, model_class:str, model_seg:str, image_box = np.index_exp[128:,64:128]):
        self._ps = ProcessSegmentation(model_seg)
        self._model_class:Model = tf.keras.models.load_model(model_class, custom_objects={'dice_coef':dice_coef, 'dice_loss':dice_loss})
        self._names = {-1: 'brak', 0:'stop',1:'przejscie dla pieszych', 2:'ograniczenie 20', 3:'ograniczenie 30', 4:'rondo'}
        self._indexes = image_box

    def getSignIfExist(self, image:np.ndarray, box_size=20, prob_th=0.5)->[int, str]:
        image = image[self._indexes]
        pack = self._ps.predictAndCrop(image, box_size, prob_th)
        if pack is None:
            return -1, self._names[-1] 
        
        im, pred = pack
        pred = self._model_class.predict(np.array([im]))[0]
        p = np.argmax(pred)
        return p, self._names[p]
        

    
def predict_multi(model_path, path):
    images = []
    dict_images = {}

    imnames = sorted(glob(path), key=lambda f: int(re.sub('\D', '', f)))
    for im_name in imnames:
        im = io.imread(im_name, as_gray=True)
        images.append(im)
    images = np.array(images)

    sp = SignPrediction('classifierMod.h5', 'modelSeg.h5')

    box_size = 20
    for i in range(len(images)):
        pred = sp.getSignIfExist(images[i])
        if pred[0] != -1:
            print(pred)
            plt.imshow(images[i], 'gray')
            plt.show()

    exit()



if __name__ == '__main__':
    #trainingCNNClassifier()
    #convert_to_dataset()
    #image_to_dataset()
    predict_multi('modelSeg.h5', 'pred/*')
    #predict('modelSeg.h5', 'topred.png')

    with open('dataset_sign_2.dat', 'rb') as f:
        dataset = pickle.load(f)

        x, y = dataset

        x = np.array(x)
        y = np.array(y)

        model = UNet2DModel()
        model.fit(x, y, batch_size=16, epochs=50)
        model.save('modelSeg.h5')

        


