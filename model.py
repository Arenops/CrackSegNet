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

def unet7(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_1')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha = 0.05)(conv1)
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_2')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha = 0.05)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv13 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv13_1')(UpSampling2D(size = (2,2))(pool1))

    conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_1')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha = 0.05)(conv2)
    conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_2')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha = 0.05)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv14 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv14_1')(UpSampling2D(size = (4,4))(pool2))

    conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_1')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha = 0.05)(conv3)
    conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_2')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha = 0.05)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv15 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv15_1')(UpSampling2D(size = (8,8))(pool3))

    conv4 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_1')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha = 0.05)(conv4)
    conv4 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_2')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha = 0.05)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv16 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv16_1')(UpSampling2D(size = (16,16))(pool4))
    conv5 = pool4
    # conv5 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_1')(pool4)
    # conv5 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_2')(conv5)
    # conv5 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_3')(conv5)
    # conv5 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_4')(conv5)

    pool5_1 = AveragePooling2D(pool_size=(32, 32))(conv5)
    conv6_1 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6_1')(pool5_1)
    pool5_2 = MaxPooling2D(pool_size=(32, 32))(conv5)
    conv6_2 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6_2')(pool5_2)
    conv6 = concatenate([conv6_1, conv6_2], axis=3)
    up6 = UpSampling2D(size = (512,512))(conv6)

    pool6_1 = AveragePooling2D(pool_size=(16, 16))(conv5)
    conv7_1 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7_1')(pool6_1)
    pool6_2 = MaxPooling2D(pool_size=(16, 16))(conv5)
    conv7_2 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7_2')(pool6_2)
    conv7 = concatenate([conv7_1, conv7_2], axis=3)
    up7 = UpSampling2D(size = (256,256))(conv7)
    
    pool7_1 = AveragePooling2D(pool_size=(8, 8))(conv5)
    conv8_1 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_1')(pool7_1)
    pool7_2 = MaxPooling2D(pool_size=(8, 8))(conv5)
    conv8_2 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_2')(pool7_2)
    conv8 = concatenate([conv8_1, conv8_2], axis=3)
    up8 = UpSampling2D(size = (128,128))(conv8)

    pool8_1 = AveragePooling2D(pool_size=(4, 4))(conv5)
    conv9_1 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_1')(pool8_1)
    pool8_2 = AveragePooling2D(pool_size=(4, 4))(conv5)
    conv9_2 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_2')(pool8_2)
    conv9 = concatenate([conv9_1, conv9_2], axis=3)
    up9 = UpSampling2D(size = (64,64))(conv9)

    merge1 = concatenate([conv13, conv14, conv15, conv16, up6, up7, up8, up9], axis=3)
    conv10 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_1')(merge1)
    conv10 = BatchNormalization()(conv10)
    conv10 = LeakyReLU(alpha = 0.05)(conv10)
    conv11 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv11_1')(conv10)
    conv12 = Conv2D(1, 1, activation = 'sigmoid', name = 'conv12_1')(conv11)

    model = Model(inputs = input, outputs = conv12)
    print('model compile')
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

# Three skip connections
def unet6(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_1')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha = 0.3)(conv1)
    conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_2')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha = 0.3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv13 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv13_1')(UpSampling2D(size = (2,2))(pool1))

    conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_1')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha = 0.3)(conv2)
    conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_2')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha = 0.3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv14 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv14_1')(UpSampling2D(size = (4,4))(pool2))

    conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_1')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha = 0.3)(conv3)
    conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_2')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha = 0.3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv15 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv15_1')(UpSampling2D(size = (8,8))(pool3))

    conv4 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_1')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha = 0.3)(conv4)
    conv4 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_2')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha = 0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv16 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv16_1')(UpSampling2D(size = (16,16))(pool4))
    conv5=pool4
    # conv5 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_1')(pool4)
    # conv5 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_2')(conv5)
    # conv5 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_3')(conv5)
    # conv5 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_4')(conv5)

    pool5 = AveragePooling2D(pool_size=(32, 32))(conv5)
    conv6 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6_1')(pool5)
    up6 = UpSampling2D(size = (512,512))(conv6)

    pool6 = AveragePooling2D(pool_size=(16, 16))(conv5)
    conv7 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7_1')(pool6)
    up7 = UpSampling2D(size = (256,256))(conv7)
    
    pool7 = AveragePooling2D(pool_size=(8, 8))(conv5)
    conv8 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_1')(pool7)
    up8 = UpSampling2D(size = (128,128))(conv8)

    pool8 = AveragePooling2D(pool_size=(4, 4))(conv5)
    conv9 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_1')(pool8)
    up9 = UpSampling2D(size = (64,64))(conv9)

    merge1 = concatenate([conv13, conv14, conv15, conv16, up6, up7, up8, up9], axis=3)
    conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_1')(merge1)
    conv10 = BatchNormalization()(conv10)
    conv10 = LeakyReLU(alpha = 0.3)(conv10)
    conv11 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv11_1')(conv10)
    conv12 = Conv2D(1, 1, activation = 'sigmoid', name = 'conv12_1')(conv11)

    model = Model(inputs = input, outputs = conv12)
    print('model compile')
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-5), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

# all layers are named
def unet5(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_1')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_2')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv12 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv12_1')(UpSampling2D(size = (2,2))(pool1))

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_1')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_2')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv13 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv13_1')(UpSampling2D(size = (4,4))(pool2))

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_1')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_2')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_1')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_2')(conv4)
    conv14 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv14_1')(UpSampling2D(size = (8,8))(conv4))
    
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_3')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_4')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_5')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_6')(conv4)
    conv4 = BatchNormalization()(conv4)

    pool4 = AveragePooling2D(pool_size=(64, 64))(conv4)
    conv5 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_1')(pool4)
    up5 = UpSampling2D(size = (512,512))(conv5)

    pool5 = AveragePooling2D(pool_size=(32, 32))(conv4)
    conv6 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv6_1')(pool5)
    up6 = UpSampling2D(size = (256,256))(conv6)
    
    pool6 = AveragePooling2D(pool_size=(16, 16))(conv4)
    conv7 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv7_1')(pool6)
    up7 = UpSampling2D(size = (128,128))(conv7)

    pool7 = AveragePooling2D(pool_size=(8, 8))(conv4)
    conv8 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv8_1')(pool7)
    up8 = UpSampling2D(size = (64,64))(conv8)

    merge1 = concatenate([conv12, conv13, conv14, up5, up6, up7, up8], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv9_1')(merge1)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv10_1')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid', name = 'conv11_1')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

#change AveragePooling to MaxPooling
def unet4(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv13 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(pool2))

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv14 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (8,8))(conv4))
    
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)

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

    merge1 = concatenate([conv13, conv14, up5, up6, up7, up8], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-5), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

def unet3s3(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv12 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(pool1))
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv13 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(pool2))

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv14 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (8,8))(conv4))
    
    conv4_1 = Conv2D(128, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv14)
    conv4_2 = Conv2D(128, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal')(conv14)
    conv4_3 = Conv2D(128, 3, activation = 'relu', dilation_rate = (8, 8), padding = 'same', kernel_initializer = 'he_normal')(conv14)
    conv4_4 = Conv2D(128, 3, activation = 'relu', dilation_rate = (16, 16), padding = 'same', kernel_initializer = 'he_normal')(conv14)
    # merge0 = concatenate([conv12, conv13, conv14, conv4_1, conv4_2, conv4_3, conv4_4], axis=3)

    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal')(conv4)

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

    merge1 = concatenate([conv12, conv13, conv14, conv4_1, conv4_2, conv4_3, conv4_4, up5, up6, up7, up8], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model.compile(optimizer = Adam(lr = 1e-6), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

# Deeplabv3MAX
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

# PSPNetMax128SK3
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
    
# PSPNetMax128SK3new
def unet3s0(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv12 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv13 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv14 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv15 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (8,8))(conv4))
    
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

    merge1 = concatenate([conv12, conv13, conv15, up5, up6, up7, up8], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model.compile(optimizer = Adam(lr = 1e-6), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model
#skip Connection + Maxpooling
def unet3s(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

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
    
    # conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (2, 2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # conv4 = Conv2D(512, 3, activation = 'relu', dilation_rate = (4, 4), padding = 'same', kernel_initializer = 'he_normal')(conv4)
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

    merge1 = concatenate([conv13, conv14, up5, up6, up7, up8], axis=3)
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

#DeeplabV3+ASPP
def unet3(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
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

    pool4 = AveragePooling2D(pool_size=(64, 64))(conv4)
    conv5 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    up5 = UpSampling2D(size = (512,512))(conv5)

    pool5 = AveragePooling2D(pool_size=(32, 32))(conv4)
    conv6 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    up6 = UpSampling2D(size = (256,256))(conv6)
    
    pool6 = AveragePooling2D(pool_size=(16, 16))(conv4)
    conv7 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool6)
    up7 = UpSampling2D(size = (128,128))(conv7)

    pool7 = AveragePooling2D(pool_size=(8, 8))(conv4)
    conv8 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool7)
    up8 = UpSampling2D(size = (64,64))(conv8)

    merge1 = concatenate([conv13, conv14, up5, up6, up7, up8], axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    # conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-6), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

#AvgPool->MaxPool
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

# PSPNet_fake128
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

#DeeplabV3+
def unet1(pretrained_weights = None, input_size = (512, 512, 3)):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv12 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(pool1))
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv13 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(pool2))
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # conv14 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (8,8))(pool3))
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv15 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (16,16))(conv5))

    # pool4 = AveragePooling2D(pool_size=(64, 64))(conv4)
    # conv5 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    # up5 = UpSampling2D(size = (512,512))(conv5)

    # pool5 = AveragePooling2D(pool_size=(32, 32))(conv4)
    # conv6 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    # up6 = UpSampling2D(size = (256,256))(conv6)
    
    # pool6 = AveragePooling2D(pool_size=(16, 16))(conv4)
    # conv7 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool6)
    # up7 = UpSampling2D(size = (128,128))(conv7)

    # pool7 = AveragePooling2D(pool_size=(8, 8))(conv4)
    # conv8 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool7)
    # up8 = UpSampling2D(size = (64,64))(conv8)

    # merge1 = concatenate([conv13, conv14, up5, up6, up7, up8], axis=3)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv15)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(inputs = input, outputs = conv11)
    print('model compile')
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        print ('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

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


