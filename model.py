import numpy as np 
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -keras.sum(alpha * keras.pow(1. - pt_1, gamma) * keras.log(pt_1)) - keras.sum((1 - alpha) * keras.pow(pt_0, gamma) * keras.log(1. - pt_0))
    return focal_loss_fixed


# Dilated Convolutions & Focal Loss
def unet3s2(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv12 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(pool1))
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv13 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(pool2))

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    # conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv14 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (8,8))(conv4))
    
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # conv4 = BatchNormalization()(conv4)

    pool4 = MaxPooling2D(pool_size=(64, 64))(conv4)
    conv5 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    up5 = UpSampling2D(size = (512,512))(conv5)

    pool5 = MaxPooling2D(pool_size=(32, 32))(conv4)
    conv6 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    up6 = UpSampling2D(size = (256,256))(conv6)
    
    pool6 = MaxPooling2D(pool_size=(16, 16))(conv4)
    conv7 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool6)
    up7 = UpSampling2D(size = (128,128))(conv7)

    pool7 = MaxPooling2D(pool_size=(8, 8))(conv4)
    conv8 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool7)
    up8 = UpSampling2D(size = (64,64))(conv8)

    merge1 = concatenate([conv12, conv13, conv14, up5, up6, up7, up8], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    # conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    # model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-5), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

# Skip Connections
def unet3s1(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv12 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(pool1))
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv13 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(pool2))

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    # conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv14 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (8,8))(conv4))

    pool4 = MaxPooling2D(pool_size=(64, 64))(conv4)
    conv5 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    up5 = UpSampling2D(size = (512,512))(conv5)

    pool5 = MaxPooling2D(pool_size=(32, 32))(conv4)
    conv6 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    up6 = UpSampling2D(size = (256,256))(conv6)
    
    pool6 = MaxPooling2D(pool_size=(16, 16))(conv4)
    conv7 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool6)
    up7 = UpSampling2D(size = (128,128))(conv7)

    pool7 = MaxPooling2D(pool_size=(8, 8))(conv4)
    conv8 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool7)
    up8 = UpSampling2D(size = (64,64))(conv8)

    merge1 = concatenate([conv12, conv13, conv14, up5, up6, up7, up8], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    # conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model.compile(optimizer = Adam(lr = 1e-6), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

# Spatial Max Pooling
def unet2s(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)#512
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv12 = UpSampling2D(size=(8, 8))(conv4)

    pool4 = MaxPooling2D(pool_size=(64, 64))(conv4)
    conv5 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    # up5 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (64,64))(conv5))
    up5 = UpSampling2D(size = (512,512))(conv5)

    pool5 = MaxPooling2D(pool_size=(32, 32))(conv4)
    conv6 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (32,32))(conv6))
    up6 = UpSampling2D(size = (256,256))(conv6)
    
    pool6 = MaxPooling2D(pool_size=(16, 16))(conv4)
    conv7 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool6)
    # up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (16,16))(conv7))
    up7 = UpSampling2D(size = (128,128))(conv7)

    pool7 = MaxPooling2D(pool_size=(8, 8))(conv4)
    conv8 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool7)
    # up8 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (8,8))(conv8))
    up8 = UpSampling2D(size = (64,64))(conv8)

    merge1 = concatenate([conv12, up5, up6, up7, up8], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

# Spatial Average Pooling
def unet2(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)#512
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv12 = UpSampling2D(size=(8, 8))(conv4)

    pool4 = AveragePooling2D(pool_size=(64, 64))(conv4)
    conv5 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    # up5 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (64,64))(conv5))
    up5 = UpSampling2D(size = (512,512))(conv5)

    pool5 = AveragePooling2D(pool_size=(32, 32))(conv4)
    conv6 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (32,32))(conv6))
    up6 = UpSampling2D(size = (256,256))(conv6)
    
    pool6 = AveragePooling2D(pool_size=(16, 16))(conv4)
    conv7 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool6)
    # up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (16,16))(conv7))
    up7 = UpSampling2D(size = (128,128))(conv7)

    pool7 = AveragePooling2D(pool_size=(8, 8))(conv4)
    conv8 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool7)
    # up8 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (8,8))(conv8))
    up8 = UpSampling2D(size = (64,64))(conv8)

    merge1 = concatenate([conv12, up5, up6, up7, up8], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

# Backbone
def unet0(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv14 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (8,8))(conv4))
    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv14)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model.compile(optimizer = Adam(lr = 1e-5), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

# U-net
def unet(pretrained_weights = None,input_size = (512,512,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)#512
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))

    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    print('model compile')
    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model.compile(optimizer = Adam(lr = 1e-5), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


