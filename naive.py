from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

irisData=load_iris()
x=irisData.data
y=irisData.target

dataset=pd.read_csv('data.csv')
x=dataset.iloc[:, -1].values
y=dataset.iloc[:, 1].values
#print(x)
#print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)
model=GaussianNB()
model.fit(x_train,y_train)
c=model.predict(x_test)
acc=accuracy_score(y_test,c)
print(c)
print("accuracy:",acc)


