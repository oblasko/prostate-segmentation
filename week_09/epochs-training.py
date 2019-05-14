import sys
import os 

epochs = sys.argv[1]

SPLIT_FOLDER = "/scratch/oblasko/mcw-prostate-split"
OUT_FOLDER = "/scratch/oblasko/improving-classification/" + str(epochs) + "-epochs-model"
TRAINING_OUT_FOLDER = "/scratch/oblasko/improving-classification/training-output/"

#importing the data
import_command = "python3 import_data.py \
        --test-mask-dir " + str(SPLIT_FOLDER) + "/test/masks \
        --test-image-dir " + str(SPLIT_FOLDER) + "/test/images \
        --train-mask-dir " + str(SPLIT_FOLDER) + "/train/masks \
        --train-image-dir " + str(SPLIT_FOLDER) + "/train/images \
        --output-dir " + str(OUT_FOLDER)

os.system(import_command)

#running each model 3 times in order to average the performance
for i in range(3):
        #running the improved classification
        train_command = "python improved_classify_images.py \
                        --input-dir " + OUT_FOLDER + \
                        " --output-dir " + OUT_FOLDER + \
                        " --num-epochs " + str(epochs) + \
                        " --validation-split 0.0"

        os.system(train_command)

        #evaluating current model and documenting it's performance
        evaluate_command = "python evaluate_classification.py  \
                        --input-dir " + OUT_FOLDER + " >> " + TRAINING_OUT_FOLDER + str(epochs) + "-epochs-model.txt"

        os.system(evaluate_command)