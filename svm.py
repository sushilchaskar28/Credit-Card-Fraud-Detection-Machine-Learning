#Import libraries
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore")

def getfscore(cm):
    TP=cm[0][0]
    TN=cm[1][1]
    FN=cm[0][1]
    FP=cm[1][0]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    fmesr=(2*precision*recall)/(precision+recall)
    return fmesr 

#load Dataset 
dataset = pd.read_csv("creditcard.csv")
colName = [i for i in dataset.columns]
colName = colName[1:-1]
dataset = dataset.sample(frac = 1) #shuffle
x = dataset.iloc[:,1:-1].values.tolist() #features
y = dataset.iloc[:,-1].values.tolist() #labels
negData = y.count(0)
posData = y.count(1)
negPer = (negData/len(y))*100 #Valid Percentage    
posPer = (posData/len(y))*100 #fraud Percentage
print("Fraud data percentage: {0}".format(posPer))
print("Dataset is highly unbalanced. We will have to balance the dataset")
x_neg = dataset.loc[dataset['Class']==0].sample(frac=1)
x_pos = dataset.loc[dataset['Class']==1].sample(frac=1)

#Geneate 5 random datasets
xnegArr=[]
for i in range(5):
    k = random.randrange(negData) - posData
    xnegArr.append(x_neg[k:k+posData])
new_dsArr=[]
for i in range(5):
    new_dsArr.append(pd.concat([x_pos, xnegArr[i]]).sample(frac = 1))
    
print("Method 3: training using Support Vector Machine....")  
print("As data is unbalanced, we will implement Weightage SVM")

ds = pd.concat([x_pos[0:350], x_neg[0:190000]]).sample(frac = 1) # training set
ds_test = pd.concat([x_pos[350:], x_neg[190000:]]).sample(frac = 1) #test set
x_train = preprocessing.scale(ds.iloc[:,1:-1].values.tolist()) 
y_train = ds.iloc[:,-1].values.tolist() 
x_test = preprocessing.scale(ds_test.iloc[:,1:-1].values.tolist())
y_test= ds_test.iloc[:,-1].values.tolist()

weights = {0:0.1, 1:10} #Initial weight array for training weights
cls_svm = LinearSVC(class_weight=weights) #Linear SVM 
cls_svm.fit(x_train, y_train)
y_pred = cls_svm.predict(x_train)
cm = confusion_matrix(y_train, y_pred)
print("For train dataset, f-measure score is: {0}".format(getfscore(cm)))

y_pred = cls_svm.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("For test dataset, f-measure score is: {0}".format(getfscore(cm)))
print("")
print("testing on complete dataset")
y_pred = cls_svm.predict(preprocessing.scale(x))
cm = confusion_matrix(y, y_pred)
print("For original dataset, f-measure score is: {0}".format(getfscore(cm)))