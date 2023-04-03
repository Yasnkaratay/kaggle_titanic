import numpy as np 
import pandas as pd 

data = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

print(data.isnull().sum())

isimler = ["yolcu_kimligi","hayatta_kaldı","psınıfı","isim","cinsiyet","yaş","sibsp","parch","bilet","ucret","kabin","binis"]
isimleri = ["yolcu_kimligi","psınıfı","isim","cinsiyet","yaş","sibsp","parch","bilet","ucret","kabin","binis"]
data.columns = isimler
data_test.columns = isimleri

print(data.head())

x = data[["psınıfı","cinsiyet","yaş","sibsp","parch"]]
y = data.iloc[:,1]
test = data_test[["psınıfı","cinsiyet","yaş","sibsp","parch"]]
print(x.isnull().sum())

x.iloc[:,2].fillna(x.iloc[:,2].median(), inplace=True)
test.iloc[:,2].fillna(test.iloc[:,2].median(), inplace=True)
print(x.isnull().sum())


from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

le = preprocessing.LabelEncoder()
x.iloc[:,1] = le.fit_transform(x.iloc[:,1])

le1 = preprocessing.LabelEncoder()
test.iloc[:,1] = le1.fit_transform(test.iloc[:,1])
"""
ohe = OneHotEncoder()
x = ohe.fit_transform(x)

ohe1 = OneHotEncoder()
test = ohe1.fit_transform(test)
"""

"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=(False))
x = sc.fit_transform(x)


y=y.values
test=test.values
from sklearn.svm import SVC

svm = SVC()
svm.fit(x, y)

svm_predict = svm.predict(test)
yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
deneme1 = pd.DataFrame(svm_predict)
deneme1.columns=["Survived"]
deneme1 = pd.concat([yolcu_kimlik,deneme1],axis=1)
deneme1.to_csv("deneme1.csv",index=False)
"""
x=x.values
y=y.values
test=test.values
"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6,weights="distance")
knn.fit(x, y)

from sklearn.model_selection import cross_val_score
cvs = cross_val_score(knn, X=x,y=y,cv=4)
print(cvs.mean())
print(cvs.std())

from sklearn.model_selection import GridSearchCV

p = [{"n_neighbors":[1,2,3,4,5,6], "weights":["uniform","distance"]}]

grid = GridSearchCV(knn, param_grid=p,n_jobs=-1,cv=10)
grid.fit(x, y)

print(grid.best_score_)
print(grid.best_params_)

knn_predict = knn.predict(test)
yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
deneme2 = pd.DataFrame(knn_predict)
deneme2.columns=["Survived"]
deneme2 = pd.concat([yolcu_kimlik,deneme2],axis=1)
deneme2.to_csv("deneme2.csv",index=False)
"""
"""
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x,y)

from sklearn.model_selection import cross_val_score
cvs = cross_val_score(dtc, X=x,y=y,cv=4)
print(cvs.mean())
print(cvs.std())

from sklearn.model_selection import GridSearchCV
p = [{"criterion":["gini","entropy","log_loss"],"splitter":["best","entropy","random"]}]
grid = GridSearchCV(dtc, param_grid=p,n_jobs=(-1),cv=10)
grid.fit(X=x,y=y)
print("-----------------")
print(grid.best_estimator_)
print(grid.best_score_)

dtc_predict = dtc.predict(test)

yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
deneme35 = pd.DataFrame(dtc_predict)
deneme35.columns=["Survived"]
deneme35 = pd.concat([yolcu_kimlik,deneme35],axis=1)
deneme35.to_csv("deneme35.csv",index=False)
"""
"""
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,criterion="gini")
rfc.fit(x, y)
rfc_predict = rfc.predict(test)

yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
deneme4 = pd.DataFrame(rfc_predict)
deneme4.columns=["Survived"]
deneme4 = pd.concat([yolcu_kimlik,deneme4],axis=1)
deneme4.to_csv("deneme4.csv",index=False)
"""
"""
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
sc.fit_transform(x)
sc1 = StandardScaler()
sc1.fit_transform(test)
 

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x, y)
lin_predict = lin.predict(test)

lin_predict = np.round(lin_predict).astype(int)

yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
deneme5 = pd.DataFrame(lin_predict)
deneme5.columns=["Survived"]
deneme5 = pd.concat([yolcu_kimlik,deneme5],axis=1)
deneme5.to_csv("deneme5.csv",index=False)
"""
"""
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(x,y)
gnb_predict = gnb.predict(test)

yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
deneme6 = pd.DataFrame(gnb_predict)
deneme6.columns=["Survived"]
deneme6 = pd.concat([yolcu_kimlik,deneme6],axis=1)
deneme6.to_csv("deneme6.csv",index=False)
"""

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
sc.fit_transform(x)
sc1 = StandardScaler()
sc1.fit_transform(test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(3, kernel_initializer='uniform', activation = 'relu' , input_dim = 5))

classifier.add(Dense(3, kernel_initializer='uniform', activation = 'relu'))

classifier.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['accuracy'] )

classifier.fit(x, y, epochs=50)

y_pred = classifier.predict(test)

y_pred = np.round(y_pred).astype(int)

yolcu_kimlik = data_test[["yolcu_kimligi"]]
yolcu_kimlik.columns=["PassengerId"]
deneme7 = pd.DataFrame(y_pred)
deneme7.columns=["Survived"]
deneme7 = pd.concat([yolcu_kimlik,deneme7],axis=1)
deneme7.to_csv("deneme7.csv",index=False)












