# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:31:05 2022
@author: Admin
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:20:58 2022
@author: Admin
"""
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score  
from sklearn import metrics
import matplotlib.pyplot as plt
#Loading datasets 
data = pd.read_csv("C://Users//LJMCA//Desktop//research paper//TISHA//Mental_illness.csv")
# Create feature and target arrays
X = data[data.columns[0:11]]

print(X)
Y = data.Anxiety
print(Y)
# Splitting the dataset into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.svm import SVC
clf = SVC(kernel = 'linear', random_state = 0)
clf.fit(X_train, y_train)
# prediction on X_test (testing data )
Y_pred=clf.predict(X_test)
print(Y_pred)
cm=np.array(confusion_matrix(y_test,Y_pred))
print(cm)
#Accuray of the model 
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))
#tree.plot_tree(clf)
plt.figure(figsize=(3,3))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 13)
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ["0", "1","2"], size = 10)
plt.yticks(tick_marks, ["0", "1","2"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 13)
plt.xlabel('Predicted label', size = 13)
width, height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x), 
  horizontalalignment='center',
  verticalalignment='center')

