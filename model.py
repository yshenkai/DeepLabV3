from keras.models import Model
from keras.layers import Input,Conv2D,DepthwiseConv2D,Activation,BatchNormalization,Add,Concatenate,ZeroPadding2D,Reshape,Lambda
from keras.layers import UpSampling2D,AveragePooling2D,Dropout
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

def sparse_crossentropy_ignoring_last_label(y_true, y_pred):

    y_true = y_true[:,:,:-1]
    return K.categorical_crossentropy(y_true, y_pred)

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    #nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, 21))
    y_true=y_true[:,:,:-1]
    y_true = K.argmax(K.reshape(y_true,(-1,21)),axis=-1)
    legal_labels = ~K.equal(y_true, 255)
    return K.sum(tf.to_float(legal_labels & K.equal(y_true,
                                                    K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))
def Jaccard(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []

    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:,:,0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

def _conv_same(x,filters,kernel_size=3,strides=1,rate=1):
    if strides==1:
        return Conv2D(filters=filters,kernel_size=kernel_size,strides=1,padding="same",use_bias=False,dilation_rate=rate)(x)
    else:
        pad_total=kernel_size+(kernel_size-1)*(rate-1)-1
        pad_beg=pad_total//2
        pad_end=pad_total-pad_beg
        x=ZeroPadding2D(padding=(pad_beg,pad_end))(x)
        return Conv2D(filters==filters,kernel_size=kernel_size,strides=strides,dilation_rate=rate,padding="valid")(x)
def _Sep_Conv_BN(x,filters,kernel_size=3,strides=1,rate=1,depth_activation=False,epilon=1e-3):
    if strides==1:
        depth_pad="same"
    else:
        pad_total=kernel_size+(kernel_size-1)*(rate-1)-1
        pad_beg=pad_total//2
        pad_end=pad_total-pad_beg
        x=ZeroPadding2D(padding=(pad_beg,pad_end))(x)
        depth_pad="valid"
    if not depth_activation:
        x=Activation("relu")(x)
    x=DepthwiseConv2D(kernel_size=kernel_size,strides=strides,dilation_rate=rate,padding=depth_pad,use_bias=False)(x)
    x=BatchNormalization(epsilon=epilon)(x)
    if depth_activation:
        x=Activation("relu")(x)
    x=Conv2D(filters=filters,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization(epsilon=epilon)(x)
    if depth_activation:
        x=Activation("relu")(x)
    return x
def _Xception_block(x,filters_list,kernel_size=3,strides=1,rate=1,depth_activation=False,epilon=1e-3,middle_Conv=True,return_skip=False):
    middle=x
    for i in range(len(filters_list)):
        x=_Sep_Conv_BN(x,filters=filters_list[i],kernel_size=kernel_size,strides=strides if i ==2 else 1,rate=rate,depth_activation=depth_activation)
        if return_skip & i==1:
            skip=x
    if middle_Conv:
        middle=_conv_same(middle,filters=filters_list[-1],kernel_size=1,strides=strides,rate=rate)
        middle=BatchNormalization(epsilon=epilon)(middle)
    x=Add()([middle,x])
    if return_skip:
        return x,skip
    else:
        return x
def DeepLabV3(weights="pascal_voc",input_tensor=None,input_shape=(256,256,3),infer=False,classes=21):
    if input_tensor is None:
        input=Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            input=Input(shape=input_shape,tensor=input_tensor)
        else:
            input=input_tensor
    #input=Input(shape=input_shape)
    x=Conv2D(filters=32,kernel_size=3,strides=2,use_bias=False)(input)
    x=BatchNormalization(epsilon=1e-5)(x)
    x=Activation("relu")(x)

    x=Conv2D(filters=64,kernel_size=3,padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)

    x=_Xception_block(x,filters_list=[128,128,128],kernel_size=3,strides=2,rate=1,depth_activation=False,middle_Conv=True)
    x,skip=_Xception_block(x,filters_list=[256,256,256],kernel_size=3,strides=2,rate=1,depth_activation=False,middle_Conv=True,return_skip=True)
    x=_Xception_block(x,filters_list=[728,728,728],kernel_size=3,strides=1,rate=1,depth_activation=False)

    for i in range(16):
        x=_Xception_block(x,filters_list=[728,728,728],kernel_size=3,strides=1,rate=1,depth_activation=False,middle_Conv=False)

    x=_Xception_block(x,filters_list=[728,1024,1024],strides=1,kernel_size=3,rate=1,depth_activation=False)
    x=Activation("relu")(x)
    x=DepthwiseConv2D(kernel_size=3,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Conv2D(filters=1536,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)

    x=DepthwiseConv2D(kernel_size=3,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Conv2D(filters=1536,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)

    x=DepthwiseConv2D(kernel_size=3,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Conv2D(filters=2048,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)

    d1=Conv2D(256,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    d1=BatchNormalization()(d1)
    d1=Activation("relu")(d1)

    d2=DepthwiseConv2D(kernel_size=3,strides=1,padding="same",dilation_rate=6,use_bias=False)(x)
    d2=BatchNormalization()(d2)
    d2=Activation("relu")(d2)
    d2=Conv2D(256,kernel_size=1,strides=1,padding="same",use_bias=False)(d2)
    d2=BatchNormalization()(d2)
    d2=Activation("relu")(d2)

    d3=DepthwiseConv2D(kernel_size=3,strides=1,padding="same",dilation_rate=12,use_bias=False)(x)
    d3=BatchNormalization()(d3)
    d3=Activation("relu")(d3)
    d3=Conv2D(256,kernel_size=1,strides=1,padding="same",use_bias=False)(d3)
    d3=BatchNormalization()(d3)
    d3=Activation("relu")(d3)

    d4=DepthwiseConv2D(kernel_size=3,strides=1,padding="same",dilation_rate=18,use_bias=False)(x)
    d4=BatchNormalization()(d4)
    d4=Activation("relu")(d4)
    d4=Conv2D(256,kernel_size=1,strides=1,padding="same",use_bias=False)(d4)
    d4=BatchNormalization()(d4)
    d4=Activation("relu")(d4)

    d5=AveragePooling2D(pool_size=(int(np.ceil(input_shape[0]/8)),int(np.ceil(input_shape[1]/8))))(x)
    d5=Conv2D(256,kernel_size=1,strides=1,padding="same",use_bias=False)(d5)
    d5=BatchNormalization()(d5)
    d5=Activation("relu")(d5)
    d5=Lambda(lambda x:K.tf.image.resize_bilinear(x,size=(int(np.ceil(input_shape[0]/8)),int(np.ceil(input_shape[1]/8)))))(x)

    x=Concatenate()([d1,d2,d3,d4,d5])
    x=Conv2D(256,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(0.1)(x)

    x=Lambda(lambda x:K.tf.image.resize_bilinear(x,size=(int(np.ceil(input_shape[0]/4)),int(np.ceil(input_shape[1]/4)))))(x)

    skip=Conv2D(filters=48,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    skip=BatchNormalization()(skip)
    skip=Activation("relu")(skip)

    x=Concatenate()([x,skip])
    x=DepthwiseConv2D(kernel_size=3,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization(epsilon=1e-5)(x)
    x=Activation("relu")(x)
    x=Conv2D(filters=256,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization(epsilon=1e-5)(x)
    x=Activation("relu")(x)

    x=DepthwiseConv2D(kernel_size=3,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization(epsilon=1e-5)(x)
    x=Activation("relu")(x)
    x=Conv2D(filters=256,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    x=BatchNormalization(epsilon=1e-5)(x)
    x=Activation('relu')(x)

    x=Conv2D(classes,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    x=Lambda(lambda x:K.tf.image.resize_bilinear(x,size=(input_shape[0],input_shape[1])))(x)
    x=Reshape((input_shape[0]*input_shape[1],classes))(x)

    x=Activation("softmax")(x)

    if input_tensor is not None:
        input=get_source_inputs(input_tensor)
    else:
        input=input
    model=Model(input,x,name="deeplabv3")
    model.compile(optimizer=Adam(lr=7e-4,epsilon=1e-8,decay=1e-6),loss=sparse_crossentropy_ignoring_last_label,metrics=[sparse_accuracy_ignoring_last_label,Jaccard])

    return model
model=DeepLabV3()
model.summary()




# def get_model():
#     input=