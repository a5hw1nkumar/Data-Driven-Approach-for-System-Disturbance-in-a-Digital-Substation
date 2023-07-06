%tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


%matplotlib inline
from google.colab import drive
drive.mount('/content/gdrive')
root_path = 'gdrive/My Drive/datas.xlsx'


#Reading the dataset from its directory
dataset=pd.read_excel(root_path)
dataset.head()

#removing the time column from the dataset
dataset.drop('time',axis=1,inplace=True)
dataset.head() 

#checkin the information present in the dataset
print("Dimensions of the dataset : ",dataset.shape[0]," rows and ",dataset.shape[1]," columns")

#standardizing the value in the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(dataset.iloc[:,:-1])
refined_dataset = sc.transform(dataset.iloc[:,:-1])
refined_dataset = pd.DataFrame(refined_dataset)
refined_dataset.head()

#Spliting the data into X and Y for training and testing purposes
y = dataset.loc[:,"faulttype_and_location"]
# finding the unique values in the predictions ie., Y
print("Number of unique value in the prediction class : ",len(np.unique(y)))
y=y.values
x = refined_dataset.values

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y)
print("Length of the training set : ",len(train_x))
print("Length of the testing set  : ",len(test_x))




#Using SVM classifier

from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(train_x,train_y)
y_pred = svm_classifier.predict(test_x)
from sklearn.metrics import accuracy_score , confusion_matrix
print("Accuracy Obtained : ",accuracy_score(test_y,y_pred)*100)


#Using Decision Tree for training

from sklearn.tree import DecisionTreeClassifier
dtc_classifier = DecisionTreeClassifier()
dtc_classifier.fit(train_x,train_y)
y_pred = dtc_classifier.predict(test_x)
print("Accuracy Obtained : ",accuracy_score(test_y,y_pred)*100)


#Training KNN Classifier

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(train_x,train_y)
y_pred = knn_classifier.predict(test_x)
print("Accuracy Obtained : ",accuracy_score(test_y,y_pred)*100)


def predict(input_value):
  preds = svm_classifier.predict([input_value])

  print("Predicted Output : ", preds)
  return preds

#Confusion Matrix for SVM classifier
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred,test_y)


plt.scatter(y_pred,test_y)
plt.scatter(y_pred1,test_y)
plt.scatter(y_pred2,test_y)
plt.legend(['SVM','Decision Tree','KNN Classifier'])

