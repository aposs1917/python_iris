import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("iris.csv")
columns=df.columns
df.describe(include=object)
df.describe()
df.isna().sum()

df['Species'].value_counts()



    

figure,axis =plt.subplots(2,2,figsize=(15,10),sharey=True)
figure.suptitle('Lengths & Widths by Species')

sns.boxplot(data=df,ax=axis[0,0],x="Species",y="PetalLengthCm")
sns.boxplot(data=df,ax=axis[0,1],x="Species",y="PetalWidthCm")
sns.boxplot(data=df,ax=axis[1,0],x="Species",y="SepalLengthCm")
sns.boxplot(data=df,ax=axis[1,1],x="Species",y="SepalWidthCm")

sns.scatterplot(data=df,x="PetalLengthCm",y="PetalWidthCm",hue="Species")

sns.scatterplot(data=df,x="SepalLengthCm",y="SepalWidthCm",hue="Species")

### Sepal complicates things between versicolor and verginica a bit


#PCA

x=df.iloc[:,1:5].to_numpy()

y=df.iloc[:,5]
y

# Preprocess / standarize

from sklearn.preprocessing import StandardScaler
x_s=StandardScaler().fit_transform(x)
x_s

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_features = pca.fit_transform(x_s)

df_pca=pd.DataFrame(
    data=pca_features,
    columns=['PC1','PC2','PC3'])


df_pca['target']=y

#informative features
fig=plt.figure(figsize=(10,10))
tot_var=sum(pca.explained_variance_)
plt.bar(x=df_pca.columns[0:3],height=pca.explained_variance_)
plt.bar(x=df_pca.columns[0:3],height=pca.explained_variance_/tot_var)
sns.scatterplot(data=df_pca,x="PC1",y="PC2",hue='target')

#most clear difference is found in setosa 


###Classification



from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
dfclass=df
dfclass['encoded']=encoder.fit_transform(dfclass['Species'])



dfclass.columns

X=dfclass.iloc[:,1:5]
y=dfclass.iloc[:,6].to_numpy()
y

from sklearn.model_selection import train_test_split
#80-20 split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,random_state=0,
    train_size=0.8)
X_train.shape


### SVM classification

scaler=StandardScaler().fit(X_train)
s_X=scaler.transform(X_train)
s_X_test=scaler.transform(X_test)

from sklearn import svm

clf=svm.SVC(kernel='poly')

clf.fit(s_X, y_train)

#Predict the response for test dataset
y_pred = clf.predict(s_X_test)

#accuracy
from sklearn import metrics

metrics.accuracy_score(y_test,y_pred)

y_pred==y_test

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
conf_mat=confusion_matrix(y_test,y_pred)

ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                 display_labels=clf.classes_).plot()

#only mismatch : predicts label 2 as label 1 --> 2 occurences
#label 1 and are

encoder.inverse_transform([1,2])

#it confuses versicolor with virginica, as we saw both from the two principal component plots and
#the sepal length vs sepal width plots , these 2 species are the most difficult to classify



#Decision tree classifier

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(max_depth=1)

# no need for preprocessing

clf=clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

metrics.accuracy_score(y_test,y_pred)

###grid search SVM

from sklearn.model_selection import GridSearchCV
svm.SVC().get_params().keys()
params={'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        'coef0':np.arange(0,1.5,0.25).tolist()}


grid=GridSearchCV(scoring="accuracy",estimator=svm.SVC(),
                  param_grid=params)

grid.fit(s_X,y_train)

print(grid.best_score_)

grid.best_estimator_
grid.best_params_

print(pd.DataFrame(grid.cv_results_))

grid.best_estimator_.get_params()

clf=svm.SVC(kernel='poly',coef0=1)

clf.fit(s_X,y_train)

y_pred_train=clf.predict(s_X)
y_pred_test=clf.predict(s_X_test)

metrics.accuracy_score(y_train,y_pred_train)
metrics.accuracy_score(y_test,y_pred_test)
