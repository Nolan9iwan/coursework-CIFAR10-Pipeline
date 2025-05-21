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

#show images from set 
plt.figure(figsize=(5, 5))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i]) 
    plt.title(Class_names[y_train[1][0]])
    plt.tight_layout()
    plt.savefig("bar_chart.png")

# check how many images per class
counts = np.bincount(y_train.flatten())

# plot bar chart to show class distribution
plt.figure(figsize=(8, 4))
sns.barplot(x=Class_names, y=counts)
plt.title("How many images per class (Training set)")
plt.ylabel("Number")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bar_chart.png")
