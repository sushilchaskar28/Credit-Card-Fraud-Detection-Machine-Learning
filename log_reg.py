#load libraries
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.decomposition import PCA
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
negPer = (negData/len(y))*100
posPer = (posData/len(y))*100
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
    
print("Method 2: training using logistic regression....")    
#training for one dataset
x_set = new_dsArr[0].iloc[:,1:-1].values.tolist()
x_set_fs= preprocessing.scale(x_set)
y_set = new_dsArr[0].iloc[:,-1].values.tolist()
x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.30, random_state=1)

cls_lr = LogisticRegression(random_state=0,solver='liblinear').fit(x_train, y_train)
y_pred = cls_lr.predict(x_test)

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("For dataset {0}, f-measure score is: {1}".format(1, getfscore(cm)))

#With feature scalling
x_train, x_test, y_train, y_test = train_test_split(x_set_fs, y_set, test_size=0.30, random_state=1)
cls_lr1 = LogisticRegression(random_state=0,solver='liblinear').fit(x_train, y_train)
y_pred = cls_lr1.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("With feature scalling, f-measure score is: {0}".format(getfscore(cm)))
print("")

print("using classifier on remaining test dataset with feature scaling")
for i in range(1,5):
    x_set = preprocessing.scale(new_dsArr[i].iloc[:,1:-1].values.tolist()) #for logistic Reg
    y_set = new_dsArr[i].iloc[:,-1].values.tolist()
    y_pred = cls_lr1.predict(x_set)
    cm = confusion_matrix(y_set, y_pred)
    print("For dataset {0}, f-measure score is: {1}".format(i+1,getfscore(cm)))

print("testing on original dataset")
y_pred = cls_lr1.predict(preprocessing.scale(x))
cm = confusion_matrix(y, y_pred)
print("For original dataset, f-measure score is: {0}".format(getfscore(cm)))
print("")

#PCA for dimentionality reduction
print("Implement PCA for dimentionality reduction...")
pca = PCA(n_components = 15)
pcaArr=[]
for i in range(5):
    d1 = pd.DataFrame(pca.fit_transform(preprocessing.scale(new_dsArr[i].iloc[:,1:-1])))
    d1.insert(15, "Class", new_dsArr[i].iloc[:,-1].values.tolist(), True) 
    pcaArr.append(d1)

convArr=np.array(pca.components_)
x_pca = pcaArr[0].iloc[:,0:-1].values.tolist()
y_pca = pcaArr[0].iloc[:,-1].values.tolist()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y_pca, test_size=0.30, random_state=1)

cls_lr_pc = LogisticRegression(random_state=0,solver='liblinear').fit(x_train, y_train)
y_pred = cls_lr_pc.predict(x_test)

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("For dataset {0}, f-measure score is: {1}".format(1, getfscore(cm)))
print("testing on original dataset")
pca_dataframe=[]
x_pre = preprocessing.scale(x)
for i in range(len(x_pre)):
    nparr=np.array([x_pre[i]])
    nparr=nparr.dot(np.transpose(convArr)).tolist()
    pca_dataframe.append([item for sublist in nparr for item in sublist])
pca_dataframe=pd.DataFrame(pca_dataframe)
pca_dataframe.insert(15, "Class", y, True) 
X_pca = pca_dataframe.iloc[:,0:-1].values.tolist()
Y_pca = pca_dataframe.iloc[:,-1].values.tolist()
y_pred = cls_lr_pc.predict(X_pca)
cm = confusion_matrix(Y_pca, y_pred)
print("For original dataset, f-measure score is: {0}".format(getfscore(cm)))

print("")
print("weighted Logistic Regression for better output")
ds = pd.concat([x_pos[0:350], x_neg[0:190000]]).sample(frac = 1)
ds_test = pd.concat([x_pos[350:], x_neg[190000:]]).sample(frac = 1)
x_train = preprocessing.scale(ds.iloc[:,1:-1].values.tolist())
y_train = ds.iloc[:,-1].values.tolist()
x_test = preprocessing.scale(ds_test.iloc[:,1:-1].values.tolist())
y_test= ds_test.iloc[:,-1].values.tolist()
#x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.30, random_state=1)
weights = {0:0.1, 1:10} #Initial weight array for weight calculations
cls_lrw = LogisticRegression(solver='saga', class_weight=weights)
cls_lrw.fit(x_train, y_train)
y_pred = cls_lrw.predict(x_train)
cm = confusion_matrix(y_train, y_pred)
print("For train dataset, f-measure score is: {0}".format(getfscore(cm)))

y_pred = cls_lrw.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("For test dataset, f-measure score is: {0}".format(getfscore(cm)))
print("")
print("testing on complete dataset")
y_pred = cls_lrw.predict(preprocessing.scale(x))
cm = confusion_matrix(y, y_pred)
print("For original dataset, f-measure score is: {0}".format(getfscore(cm)))
