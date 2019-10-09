from model import DeepLabV3
import numpy as np
import os
import multiprocessing
works=multiprocessing.cpu_count()//2
import keras
from keras.utils.data_utils import Sequence
import tensorflow as tf
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,LambdaCallback
from keras.utils import to_categorical
from sklearn.utils import class_weight
import cv2
import random
import numbers
import pydensecrf.densecrf as dcrf
from keras import backend as K
import itertools

def get_VOC2012_classes():
    PASCAL_VOC_classes = {
        0: 'background',
        1: 'airplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'table',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'potted_plant',
        17: 'sheep',
        18: 'sofa',
        19 : 'train',
        20 : 'tv',
        21 : 'void'
    }
    return PASCAL_VOC_classes
def sparse_crossentropy_ignoring_last_label(y_true,y_pred):
    nb_classes=K.int_shape(y_pred)[-1]
    y_true=K.one_hot(tf.to_int32(y_true[:,:,0]),nb_classes+1)[:,:,:-1]
    return K.categorical_crossentropy(y_true,y_pred)
def spares_accuracy_ignoring_lastr_label(y_true,y_pred):
    nb_classes=K.int_shape(y_pred)[-1]
    y_pred=K.reshape(y_pred,(-1,nb_classes))
    y_true=tf.to_int32(K.flatten(y_true))
    legal_label=~K.equal(y_true,nb_classes)
    return K.sum(tf.to_float(legal_label&K.equal(y_true,K.argmax(y_pred,axis=-1))))/K.sum(tf.to_float(legal_label))

def compute_miou(y_true,y_pred):
    nb_classes=K.int_shape(y_pred)[-1]
    iou=[]
    pred_pixels=K.argmax(y_pred,axis=-1)
    for i in range(0,nb_classes):
        true_label=K.equal(y_true[:,:,0],i)
        pred_label=K.equal(pred_pixels,i)
        inter=tf.to_int32(true_label&pred_label)
        union=tf.to_int32(true_label | pred_label)
        legal_batches=K.sum(tf.to_int32(true_label),axis=1)>0
        ious=K.sum(inter,axis=1)/K.sum(union,axis=1)
        iou.append(K.mean(tf.gather(ious,indices=tf.where(legal_batches))))
    iou=tf.stack(iou)
    legal_labels=~tf.debugging.is_nan(iou)
    iou=tf.gather(iou,indices=tf.where(legal_labels))
    return K.mean(iou)
