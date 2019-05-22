# Week 9 report

In the week 9 I focused on the optimization of reinforcement classification model from the last week. In order to automate the workflow I created script that each time runs the training of the model with different parameters.

```
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
```
## Training the transfer learning classification on different number of epochs

Each number of epochs were run 3 times and then averaged.

|  Number of epochs |  Accuracy  |
|---|---|
| 5  | 0.89  |   
 10  | 0.88  |  
| 15  | 0.89   | 
| 20  | 0.90  |  
| 25  | 0.89   | 
| 30  | 0,89  |  
| 35  | 0,89  |  
| 40  | 0,89  |  
| 45  | 0,89  |  
| 50  | 0,89  |  

As we can see the optimal number of epochs seems to be 20 with 0.90 accuracy.

## Training the transfer learning classification on different number of nodes in dense layers

|  Number of nodes in dense layers |  Accuracy  |
|---|---|
| 128 - 128 - 1  | 0.88  |   
| 256 - 256 - 1  | 0.89  |   
| 256 - 128 - 1  | 0.89  |   
| 512 - 512 - 1  |  X - GPU memory exceeded | 
| 512 - 256  |  X - GPU memory exceeded | 

The number of nodes in the dense layers doesn't seem to affect the accuracy much, however the optimal number seems to be 256-128-1 respectively.

## Training the transfer learning classification with a different dropout rates

We added a dropout layer after the top_layer of the segmentation model. However it didn't improve the accuracy and with growing dropout rate the accuracy was decreasing.

|  Dropout rate |  Accuracy  |
|---|---|
| 0.2  | 0.88  |   
| 0.6  | 0.88   |   
| 0.8   | 0.81   |   


## Starting from different layer

We tried appending the dense layers to a different convolutional layer of the segmentation model in order to see if it affects the performance. The differences weren't significant but the 13th layer seemed like the best option.

| Starting layer |  Accuracy  | 
|---|---|
| 10th  |  X - GPU memory exceeded |    
| 11th  |  X - GPU memory exceeded | 
| 12th  | 0.88   |   
| 13th  |  0.90 |   
| 14th  | 0.89  |   

## The transfer learning classification model with optimal parameters

**Base classification model metrics**
Accuracy: 0.891
Loss: 0.386

**Optimized transfer learning classification model**
Accuracy: 0.891
Loss: 0.378
