import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "6" #The number 6 indicates the cores of the CPU


def knn(k):
    #setting up a knn model with given k    
    knn = KNeighborsClassifier(n_neighbors=k)

    # Training our model
    knn.fit(x_train_flat, y_train_flat)

    # Make predictions on the test dataset
    y_pred = knn.predict(x_test_flat)

    # Calculating the accuracy of knn
    accuracy = accuracy_score(y_test_flat, y_pred)
    print(f"Accuracy of {k}-NN:", accuracy*100, "%")

def ncc():
    #setting up a ncc model   
    clf = NearestCentroid()
    
    # Training our model
    clf.fit(x_train_flat, y_train_flat)
    # Make predictions on the test dataset
    y_pred = clf.predict(x_test_flat)
    
    # Calculating the accuracy of knn
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of NCC:", accuracy*100, "%")

 
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)
print(x_train_flat.shape)
# Reshape labels to 1D
y_train_flat = y_train.flatten()
y_test_flat = y_test.flatten()
print(y_train_flat.shape)
knn(1) #running knn for k = 1
knn(3) #running knn for k = 3
ncc() #running ncc




