import tensorflow 
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = cifar10.load_data() #Loading Cifar10 dataset from tensorflow libraries

x_train = x_train / 255.0  # Normalize x_train elements to get values from 0 to 1
y_train = np.eye(10)[y_train.squeeze()] # Convert y_train to one hot decoding

x_test = x_test / 255.0  # Normalize x_test elements to get values from 0 to 1
y_test = np.eye(10)[y_test.squeeze()] # Convert y_testto one hot decoding

# Setting up our data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)   
datagen.fit(x_train)

# Define our model layers
Conv_1 = Conv2D(filters=32, kernel_size=(2, 2), activation=LeakyReLU(alpha=0.01),trainable=True, use_bias=True, padding = "same")
Conv_2 = Conv2D(filters=128, kernel_size=(4, 4), activation=LeakyReLU(alpha=0.01),trainable=True, use_bias=True, padding = "same")
Conv_3 = Conv2D(filters=256, kernel_size=(3, 3), activation=LeakyReLU(alpha=0.01),trainable=True, use_bias=True, padding = "same")
Pool_a = MaxPooling2D(3, 3)
Pool_b = MaxPooling2D(2,2)
flat = Flatten()
Output_class = Dense(10, activation='softmax')

# Setting up our CNN model
CNN_model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv_1,
    Conv_2,
    Dropout(0.2),
    Pool_a,
    Conv_3,
    Dropout(0.1),
    Pool_b,
    flat,
    Output_class
])

optimizer = Adam(learning_rate=0.001) #Setting up our optimizer and the initial learning rate
CNN_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy']) #Compiling the code



CNN_model.summary() # An overbiew of our model

# Using a function to create dynamic learning rate
def variable_learning_rate(epoch, lr):
    if epoch <= 7:
        return lr  
    else:
        return lr * 0.95

lr_scheduler = LearningRateScheduler(variable_learning_rate)

# Running the training and testing process
history = CNN_model.fit(datagen.flow(x_train, y_train, batch_size=128), validation_data=(x_test, y_test), epochs=20, callbacks=[lr_scheduler])

test_loss, test_acc = CNN_model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}") #Printing the final testing accuracy of our model
