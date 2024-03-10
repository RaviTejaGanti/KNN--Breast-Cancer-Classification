import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
df = pd.read_csv('C:/Users/ravit/Documents/ML/Supervised Learning Algorithms/KNN/breast-cancer-wisconsin.data.txt')
df.head(5)
col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion', 
             'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

df.columns = col_names

df.columns
df['Class'].value_counts()
df.drop(['Id'],axis=1,inplace=True)
df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'],errors='coerce')
df.dtypes

# Splitting to Features and Target Variables to train and test
X = df.drop(['Class'],axis=1)
y = df['Class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_test.shape)
print(X_train.shape)

# Feature engineering 
X_train.isnull().sum()
X_test.isnull().sum()
X_train['Bare_Nuclei'].value_counts()
X_train['Bare_Nuclei']=X_train['Bare_Nuclei'].fillna(X_train['Bare_Nuclei'].median())
X_train.isnull().sum()
X_test['Bare_Nuclei'] = X_test['Bare_Nuclei'].fillna(X_train['Bare_Nuclei'].median())
X_test.isnull().sum()


# Developing a KNN model 
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

#Predicting the probabilities of predictions for class 0(2) and class 1(4) 
knn.predict_proba(X_test)[:,0]
knn.predict_proba(X_test)[:,1]



#finding accuracy of the model on test and predictions
print('Accuracy of the model with k=3 is {0:0.4f}'.format(accuracy_score(y_test,y_pred)))
#Finding the train accuracy of the mode
y_train_pred = knn.predict(X_train)
print('Accuracy of the model with k=3 is {0:0.4f}'.format(accuracy_score(y_train_pred,y_train)))


# Testing with different K-values
# k = 5
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(X_train,y_train)
y_pred = knn_5.predict(X_test)
print('Accuracy of the model with k=5 is {0:0.4f}'.format(accuracy_score(y_test,y_pred)))
y_train_pred = knn_5.predict(X_train)
print('Accuracy of the model with k=5 is {0:0.4f}'.format(accuracy_score(y_train,y_train_pred)))

#k=6
knn_6 = KNeighborsClassifier(n_neighbors=6)
knn_6.fit(X_train,y_train)
y_pred = knn_6.predict(X_test)
print('Accuracy of the model with k=6 is {0:0.4f}'.format(accuracy_score(y_test,y_pred)))
y_train_pred = knn_6.predict(X_train)
print('Accuracy of the model with k=6 is {0:0.4f}'.format(accuracy_score(y_train,y_train_pred)))

#Looping everything
k=0
k_values = []
acc_score = []
for k in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    k_values.append(k)
    acc_score.append(accuracy_score(y_pred,y_test))

len(acc_score)
len(k_values)
k_values
acc_score
output = pd.DataFrame(k_values,acc_score)
sns.lineplot(data=output,x=k_values,y=acc_score)
plt.show()


cv_values = []
b_values = []
error_rate = []
# Cross Validation scores with K 
for b in range(1,50):
    b_nn = KNeighborsClassifier(n_neighbors=b)
    b_nn.fit(X_train,y_train)
    y_pred = b_nn.predict(X_test)
    cv_scores = cross_val_score(b_nn,X_train,y_train,cv=10,scoring='accuracy')
    cv_values.append(cv_scores.mean())
    error_value = 1-cv_scores.mean()
    error_rate.append(error_value)
    #print(cv_values)
    b_values.append(b)
    #print(b_values)
    

len(cv_values)
len(b_values)
cv_dataframe = pd.DataFrame(b_values,cv_values)
sns.lineplot(data=cv_dataframe,x=b_values,y=cv_values)
cv_dataframe_2 = pd.DataFrame(b_values,error_rate)
sns.lineplot(data=cv_dataframe,x=b_values,y=error_rate)
plt.show()