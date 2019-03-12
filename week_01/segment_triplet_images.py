from __future__ import print_function

import argparse

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import plot_model
    
from data import load_train_data, load_test_data,  load_train_data_with_flnames

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.

#extracts the patient id from the filename of the image
def getPatientId(filename):
    split_filename = filename.split('_')
    return split_filename[1]

#extracts the slice id from the filename of the image
def getSliceId(filename):
    split_filename = filename.split('_')
    return split_filename[3][:-4]

#groups all corresponding image indexes by patient
def split_images_by_patient(imgs, masks, flnames):
    indexes_by_patient = {}
    for i in range(len(imgs)):
        patient_id = getPatientId(flnames[i])
        if patient_id in indexes_by_patient:
            indexes_by_patient[patient_id].append((i, flnames[i]))
        else:
            indexes_by_patient[patient_id] = [(i, flnames[i])]
    return indexes_by_patient

#sorting criteria
def get_key(item):
    return int(getSliceId(item[1]))

#sorts the filenames and indexes by slice id
def sort_by_slice(dict):
    for id, flnames in dict.items():
        dict[id] = sorted(flnames, key=get_key)
    return dict

#concates adjacent images in the following order: 1-2-3, 2-3-4, 3-4-5, 4-5-6, ... and generates labels from masks
def create_concated_imgs(imgs, flnames, masks):
    concated_imgs = []
    labels = []
    black_img = np.zeros(shape=(512,512,1))
    for id, flname_list in flnames.items():
        for i in range(len(flname_list)-1):
            if i == 0:
                sec_img = imgs[flname_list[i][0]]
                thrd_img = imgs[flname_list[i+1][0]]
                concated_imgs.append(np.concatenate( (black_img, sec_img, thrd_img), axis=2))
                labels.append( masks[ flname_list[i][0] ].flatten().max() > 0)
    
            if i + 2 >= len(flname_list):
                first_img = imgs[flname_list[i][0]]
                sec_img = imgs[flname_list[i+1][0]]
                concated_imgs.append(np.concatenate( (first_img, sec_img, black_img), axis=2))
                labels.append( masks[ flname_list[i+1][0] ].flatten().max() > 0)
            else:    
                first_img = imgs[flname_list[i][0]]
                sec_img = imgs[flname_list[i+1][0]]
                thrd_img = imgs[flname_list[i+2][0]]
                concated_imgs.append(np.concatenate( (first_img, sec_img, thrd_img), axis=2))
                labels.append( masks[ flname_list[i+1][0] ].flatten().max() > 0)
    return np.array(concated_imgs), np.array(labels)      

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def np_dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet(loss_type):
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    if loss_type == "dice":
        loss = dice_coef_loss
    else:
        loss = "binary_crossentropy"

    model.compile(optimizer=Adam(lr=1e-5),
                  loss=loss,
                  metrics=["accuracy", dice_coef])

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

    #load images and masks as numpy arrays (N_images, 512, 512) and filenames as numpy array (N_images)
    imgs_train, imgs_mask_train, imgs_flnames_train = load_train_data_with_flnames(args.input_dir)

    #change shape of images and masks to (N_images, 512, 512, 1)
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    #get dictionary of patient ids and corresponding indexes of images and masks
    images_by_patient_train = split_images_by_patient(imgs_train, imgs_mask_train, imgs_flnames_train)
    
    #sort the splitted images by slice id
    sorted_flnames_train = sort_by_slice(images_by_patient_train)
   
    #concate neighbours and generate labels
    imgs_train, train_labels = create_concated_imgs(imgs_train, sorted_flnames_train, imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet(args.loss)

    #plot_model(model, to_file='segmentation_triplet_model.png')
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
 #   model.fit(imgs_train,
 #             imgs_mask_train,
 #             batch_size=args.batch_size,
 #             epochs=args.num_epochs,
 #             verbose=1,
 #             shuffle=True,
 #             validation_split=args.validation_split)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    #load images and masks as numpy arrays (N_images, 512, 512) and filenames as numpy array (N_images)
    imgs_test, imgs_mask_test, imgs_flname_test = load_test_data(args.input_dir)

    #change shape of images and masks to (N_images, 512, 512, 1)
    imgs_test = preprocess(imgs_test)
    imgs_mask_test = preprocess(imgs_mask_test)
    
    #get dictionary of patient ids and corresponding indexes of images and masks
    images_by_patient_test = split_images_by_patient(imgs_test, imgs_mask_test, imgs_flname_test)
    
    #sort the splitted images by slice id
    sorted_flnames_test= sort_by_slice(images_by_patient_test)
    
    #concate neighbours and generate labels
    imgs_test, test_labels = create_concated_imgs(imgs_test, sorted_flnames_test, imgs_mask_test)
   

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]

    print("Test images shape: " + str(imgs_train.shape) )
    print("Test masks shape: " + str(imgs_mask_train.shape) )
    print(sorted_flnames_test)
    print()
    print(test_labels)
    return

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_pred_mask_test = model.predict(imgs_test,
                                        verbose=1,
                                        batch_size = args.batch_size)

    imgs_pred_mask_train = model.predict(imgs_train,
                                         verbose=1,
                                         batch_size = args.batch_size)
    
    # Predictions need to be thresholded
    binary = np.zeros(imgs_pred_mask_test.shape, dtype=np.uint8)
    binary[imgs_pred_mask_test > 0.5] = 255
    imgs_pred_mask_test = binary

    binary_train = np.zeros(imgs_pred_mask_train.shape, dtype=np.uint8)
    binary_train[imgs_pred_mask_train > 0.5] = 255
    imgs_pred_mask_train = binary_train
    
    np.save(os.path.join(args.output_dir, 'imgs_pred_mask_test.npy'),
            imgs_pred_mask_test)

    np.save(os.path.join(args.output_dir, 'imgs_pred_mask_train.npy'),
            imgs_pred_mask_train)
    
    scaled = binary / 255.
    n_test_images = imgs_pred_mask_test.shape[0]
    test_dice = np.zeros(n_test_images)

    for i in xrange(n_test_images):
        d = np_dice_coef(imgs_mask_test[i],
                         scaled[i])
        test_dice[i] = d

    np.save(os.path.join(args.output_dir, 'imgs_pred_mask_test_dice.npy'),
            test_dice)
    
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    for image, image_id in zip(binary, imgs_flname_test):
        image = image[:, :, 0]
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

    print('-'*30)
    print('Evaluating model on test data...')
    print('-'*30)
    loss, accuracy, dice = model.evaluate(imgs_test,
                                          imgs_mask_test,
                                          batch_size = args.batch_size)

    print("Loss:", loss)
    print("Accuracy:", accuracy)
    print("Dice coefficient:", dice)

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-epochs",
                        type=int,
                        required=True)

    parser.add_argument("--batch-size",
                        type=int,
                        default=8)

    parser.add_argument("--validation-split",
                        type=float,
                        default=0.0)

    parser.add_argument("--input-dir",
                        type=str,
                        required=True)

    parser.add_argument("--output-dir",
                        type=str,
                        required=True)

    parser.add_argument("--loss",
                        type=str,
                        choices=["crossentropy",
                                 "dice"],
                        default="dice")
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parseargs()
    train_and_predict(args)
