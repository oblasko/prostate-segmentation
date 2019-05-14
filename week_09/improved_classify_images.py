from __future__ import print_function

import argparse

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D, BatchNormalization, Dropout, ReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.layers import *
from keras.models import load_model

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def get_unet(dropout_rate):
    #load the U-net segmentation model
    seg_model = load_model('improving-classification-segmentaion.h5', custom_objects={'dice_coef': dice_coef})
    
    #get the weights and biases of the 14th layer -- the last convolutional layer before transposation -- conv2d_10
    #weights, biases = seg_model.layers[14].get_weights()

    # freezing the weights
    for i in xrange(14):
        seg_model.layers[i].trainable = False

    top_layer = seg_model.layers[13].output
    
    #inputs = Input((img_rows, img_cols, 1))


    #we will load the trained weights into this layer
    #conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(top_layer)
    
    flatten = Flatten()(top_layer)
    dense1 = Dense(256, activation='relu')(flatten)
    dense2 = Dense(128, activation='relu')(dense1)
    dense3 = Dense(1, activation='sigmoid')(dense2)

    model = Model(inputs=[seg_model.layers[0].input], outputs=[dense3])

    loss = "binary_crossentropy"
    model.compile(optimizer=Adam(lr=1e-5),
                  loss=loss,
                  metrics=["accuracy"])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]

    return imgs_p


def train_and_predict(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pred_dir = os.path.join(args.output_dir,
                            "preds")
        
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data(args.input_dir)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    train_labels = [mask_img.flatten().max() > 0 \
                    for mask_img in imgs_mask_train]
    train_labels = np.array(train_labels)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet(args.dropout_rate)
    
    #print(model.summary())
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train,
              train_labels,
              batch_size=args.batch_size,
              epochs=args.num_epochs,
              verbose=1,
              shuffle=True,
              validation_split=args.validation_split)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_mask_test, imgs_flname_test = load_test_data(args.input_dir)
    imgs_test = preprocess(imgs_test)
    imgs_mask_test = preprocess(imgs_mask_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]
    test_labels = [mask_img.flatten().max() > 0 \
                   for mask_img in imgs_mask_test]
    test_labels = np.array(test_labels)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    pred_labels = model.predict(imgs_test,
                                verbose=1,
                                batch_size = args.batch_size)

    # Predictions need to be thresholded
    binary = np.zeros(pred_labels.shape)
    binary[pred_labels > 0.5] = 1.0

    np.save(os.path.join(args.output_dir, 'pred_image_classes.npy'),
            binary)

    print('-'*30)
    print('Evaluating model on test data...')
    print('-'*30)
    loss, accuracy = model.evaluate(imgs_test,
                                    test_labels,
                                    batch_size = args.batch_size)

    print("Loss:", loss)
    print("Accuracy:", accuracy)

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-epochs",
                        type=int,
                        required=True)

    parser.add_argument("--batch-size",
                        type=int,
                        default=8)

    parser.add_argument("--dropout-rate",
                        type=float,
                        default=0.75)
    
    parser.add_argument("--validation-split",
                        type=float,
                        default=0.2)

    parser.add_argument("--input-dir",
                        type=str,
                        required=True)

    parser.add_argument("--output-dir",
                        type=str,
                        required=True)

    return parser.parse_args()
    
if __name__ == '__main__':
    args = parseargs()
    train_and_predict(args)