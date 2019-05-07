# Week 8 report

### Training the U-net segmentation model and saving it
```
model.save('improving-classification-segmentaion.h5')
```

### Loading the model from .h5 file with a custom metric
```
#load the U-net segmentation model
seg_model = load_model('improving-classification-segmentaion.h5', custom_objects={'dice_coef': dice_coef})
```

### Loading the weights of the last convolutional layer
```
#get the weights and biases of the 14th layer -- the last convolutional layer before transposation -- conv2d_10
weights, biases = seg_model.layers[14].get_weights()
```

### Model architecture
```
inputs = Input((img_rows, img_cols, 1))

#we will load the trained weights into this layer
conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(inputs)

dense1 = Dense(512)(conv1)
dense2 = Dense(256, activation='relu')(dense1) 
dense3 = Dense(1, activation='sigmoid')(dense2)

model = Model(inputs=[inputs], outputs=[dense3])
```

### Setting the loaded weights from segmentation model and freezing them
```
#setting the weights
model.layers[1].set_weights([weights,biases])

#freezing the weights 
model.layers[1].trainable = False
```

### Problem -- not compatible shape of weights
```
Layer weight shape (3, 3, 1, 512) not compatible with provided weight shape (3, 3, 512, 512)
```

#### This is how the layers look like
```
# segmentation model layer we get the weights from
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

# improved classification layer we load the weights into
conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(inputs)
```