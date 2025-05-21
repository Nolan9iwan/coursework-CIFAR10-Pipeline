from tensorflow.keras.datasets import cifar10 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns

#load data set 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#class names
Class_names = ['airplane', 'autombile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Print examples 
print("example labels and their class names:")
for i in range (10): 
    label = y_train[i][0]
    print("Label:", label, "->", Class_names[label])

