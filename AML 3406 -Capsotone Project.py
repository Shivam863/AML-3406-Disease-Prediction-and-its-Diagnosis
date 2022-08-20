import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time
import pickle

training_data= pd.read_csv(r"C:\Users\kumsh\Desktop\AML Lectures\AML 3406\Presentation\Diseases dataset\Training.csv")
cor_matrix= training_data.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.80)]


training_data1 = training_data.drop(to_drop, axis=1)

x= training_data1.loc[:,training_data1.columns != "Diseases"]
y=training_data1.loc[:,"Diseases"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
sc= StandardScaler()
sc.fit(x_train)
x_train= sc.transform(x_train)
sc.fit(x_test)
x_test= sc.transform(x_test)


knn = KNeighborsClassifier(n_neighbors = 19)  # k = 
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)

print(len(prediction))
pickle.dump(knn,open('Disease Prediction.pkl','wb'))

model = pickle.load(open('Disease Prediction.pkl','rb'))
print(model.predict([[1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]))



