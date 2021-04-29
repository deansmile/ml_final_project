# Importing the libraries to be used:
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import warnings
def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def logreg_model(c , X_train, Y_train, X_test, Y_test):
    # Create an object of logistic regression model using linear_model.
    # Pass the value of penalty as 'L1'. By default, it is 'L2'.
    # Pass the value of C = c. Note that C is the inverse of lambda. So, small value of C i.e. b/w 0 and 1 
    # means stronger regularization and large value means less regularization.
    # Also, in sklearn, L1 is only supported with solver = 'saga'. Solver is the type of optimization algorithm like GDA or
    # SGDA, which is to be used. So, 'saga' is another algorithm like that. Pass the value of solver as 'saga'

    # TODO - Create the Logistic Regression model object as described above and save it to logreg - 5 points
    logreg = linear_model.LogisticRegression(penalty='l1',C=c,solver='saga')
    
    # TODO - Fit the model on the training set - 5 points
    logreg.fit(X_train, Y_train)
    
    # TODO - Find the prediction on training set - 5 points
    Yhat_train = logreg.predict(X_train)
    
    # Adding training accuracy to acc_train_logreg
    acc_train = logreg.score(X_train, Y_train)
    acc_train_logreg.append(acc_train)
    print("Accuracy on training data = %f" % acc_train)
    
    # TODO - Find the prediction on test set - 5 points
    Yhat_test = logreg.predict(X_test)
    
    # Adding testing accuracy to acc_test_logreg
    acc_test = logreg.score(X_test, Y_test)
    acc_test_logreg.append(acc_test)
    print("Accuracy on test data = %f" % acc_test)
    
    # Appending value of c for graphing purposes
    c_logreg.append(c)

def logreg2_model(c , X_train, Y_train, X_test, Y_test):
    # Create an object of logistic regression model using linear_model.
    # Pass the value of C=c.
    # You need not pass other parameters as penalty is 'L2' by default.
    
    # TODO - Create the Logistic Regression model object as described above and save it to logreg2 - 5 points
    logreg2 = linear_model.LogisticRegression(C=c)
    
    # TODO - Fit the model on the training set - 5 points
    logreg2.fit(X_train, Y_train)
    
    # TODO - Find the prediction on training set - 5 points
    Yhat_train = logreg2.predict(X_train)
    
    # Adding training accuracy to acc_train_logreg2
    acc_train = logreg2.score(X_train, Y_train)
    acc_train_logreg2.append(acc_train)
    print("Accuracy on training data = %f" % acc_train)
    
    # TODO - Find the prediction on test set - 5 points
    Yhat_test = logreg2.predict(X_test)
    
    # Adding testing accuracy to acc_test_logreg2
    acc_test = logreg2.score(X_test, Y_test)
    acc_test_logreg2.append(acc_test)
    print("Accuracy on test data = %f" % acc_test)
    
    # Appending value of c for graphing purposes
    c_logreg2.append(c)

allFiles1=getListOfFiles('/data_1/un/mlproject/data/Charlock')
allFiles2=getListOfFiles('/data_1/un/mlproject/data/Common Chickweed')
allFiles3=getListOfFiles('/data_1/un/mlproject/data/Shepherd’s Purse')
allFiles=allFiles1+allFiles2+allFiles3
allFiles.sort()
min_w=30
min_h=30

X = []
y = []
for i in range(len(allFiles)):
  str_y=allFiles[i].split('/')[-2]
  if str_y=='Charlock':
    y.append(0)
  elif str_y=='Common Chickweed':
    y.append(1)
  elif str_y=='Shepherd’s Purse':
    y.append(2)
  img=cv2.imread(allFiles[i])
  img=cv2.resize(img,(min_w,min_h))
  green = [0 for i in range(min_w*min_h)]
  for r in range(img.shape[0]):
    for c in range(img.shape[1]):
      pb=img[r][c][0]
      pg=img[r][c][1]
      pr=img[r][c][2]
      green[r*min_h+c]=2*pg-pr-pb
  X.append(green)
print(len(allFiles),len(X),len(y))

X_scale = StandardScaler()
X = X_scale.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
'''
np.save( 'X_train.npy' , X_train )
np.save( 'Y_train.npy' , Y_train )
np.save( 'X_test.npy', X_test)
np.save( 'Y_test.npy', Y_test)


X_train=np.load('X_train.npy')
Y_train=np.load('Y_train.npy')
X_test=np.load('X_test.npy')
Y_test=np.load('Y_test.npy')


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)

acc_train_logreg = []
acc_test_logreg = []
c_logreg = []


cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10]
for c in cVals:
    logreg_model(c, X_train, Y_train, X_test, Y_test)

  
warnings.filterwarnings("ignore")
acc_train_logreg=[]
acc_test_logreg=[]
c_logreg=[]
c_values=[]
for i in range(6):
    for j in range(1,10):
        c_values.append((10**i)*j)
for c in c_values:
    logreg_model(c/10000, X_train, Y_train, X_test, Y_test)
plt.plot(c_logreg,acc_train_logreg,'.-',color='green')
plt.plot(c_logreg,acc_test_logreg,'.-',color='red')
# Use the following function to have a legend
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
plt.grid()
plt.savefig('log_lasso.png', bbox_inches='tight')



acc_train_logreg2 = []
acc_test_logreg2 = []
c_logreg2 = []

import warnings
warnings.filterwarnings("ignore")
acc_train_logreg2=[]
acc_test_logreg2=[]
c_logreg2=[]
c_values=[]
for i in range(6):
    for j in range(1,10):
        c_values.append((10**i)*j)
for c in c_values:
    logreg2_model(c/10000, X_train, Y_train, X_test, Y_test)
plt.plot(c_logreg2,acc_train_logreg2,'.-',color='green')
plt.plot(c_logreg2,acc_test_logreg2,'.-',color='red')

# Use the following function to have a legend
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='upper right')
plt.grid()
plt.savefig('log_ridge.png', bbox_inches='tight')
'''

poly = PolynomialFeatures(2)
X_transformed_train = poly.fit_transform(X_train)
X_transformed_test = poly.fit_transform(X_test)
print(X_transformed_train.shape)
print(X_transformed_test.shape)
acc_train_logreg = []
acc_test_logreg = []
c_logreg = []
 
import warnings
warnings.filterwarnings("ignore")
acc_train_logreg=[]
acc_test_logreg=[]
c_logreg=[]
c_values=[]
for i in range(6):
    for j in range(1,10):
        c_values.append((10**i)*j)
for c in tqdm(c_values):
    logreg_model(c/10000, X_transformed_train, Y_train, X_transformed_test, Y_test)
plt.plot(c_logreg,acc_train_logreg,'.-',color='green')
plt.plot(c_logreg,acc_test_logreg,'.-',color='red')
# Use the following function to have a legend
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
plt.grid()
plt.savefig('log_poly_lasso.png', bbox_inches='tight')
ct=[]
ct1=[]
for i in range(len(c_logreg)):
    ct.append((c_logreg[i],acc_test_logreg[i]))
    ct1.append((c_logreg[i],acc_train_logreg[i]))
fo=open('log_poly_lasso.txt','w')
fo.write(str(ct)+'\n')
fo.write(str(ct1)+'\n')
'''
acc_train_svm_linear = []
acc_test_svm_linear = []
c_svm_linear = []
from sklearn import svm

# Complete the function below:
# In this function and next 2 functions, we are not passing the data matrices as parameters 
# because we can use global variables inside the functions.
def svm_linear(c):
    # TODO - Create an object of svm.SVC(probability = False, kernel = 'linear', C = c) - 5 points
    svc_linear = svm.SVC(probability = False, kernel = 'linear', C = c)
    
    # TODO - Fit the classifier on the training set - 5 points
    svc_linear.fit(X_train,Y_train)
    
    # TODO - Find the prediction and accuracy on the training set - 5 points
    Yhat_svc_linear_train = svc_linear.predict(X_train)
    acc_train = svc_linear.score(X_train, Y_train)
    
    # Adding testing accuracy to acc_train_svm
    acc_train_svm_linear.append(acc_train)
    #print('Train Accuracy = {0:f}'.format(acc_train))
    
    # TODO - Find the prediction and accuracy on the test set - 5 points
    Yhat_svc_linear_test = svc_linear.predict(X_test)
    acc_test = svc_linear.score(X_test, Y_test)
    
    # Adding testing accuracy to acc_test_svm
    acc_test_svm_linear.append(acc_test)
    #print('Test Accuracy = {0:f}'.format(acc_test))
    
    # Appending value of c for graphing purposes
    c_svm_linear.append(c)
    
import warnings
warnings.filterwarnings("ignore")
acc_train_svm_linear=[]
acc_test_svm_linear=[]
c_svm_linear=[]
c_values=[]
for i in range(5):
    for j in range(1,10):
        c_values.append((10**i)*j)
for c in c_values:
    svm_linear(c/10000)
plt.plot(c_svm_linear,acc_train_svm_linear,'.-',color='green')
plt.plot(c_svm_linear,acc_test_svm_linear,'.-',color='red')


# Use the following function to have a legend
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
plt.grid()
plt.savefig('svm_linear.png', bbox_inches='tight')

acc_train_svm_rbf = []
acc_test_svm_rbf = []
c_svm_rbf = []
from sklearn import svm

# Complete the function below:
# In this function and next 2 functions, we are not passing the data matrices as parameters 
# because we can use global variables inside the functions.
def svm_rbf(c):
    # TODO - Create an object of svm.SVC(probability = False, kernel = 'rbf', C = c) - 5 points
    svc_rbf = svm.SVC(probability = False, kernel = 'rbf', C = c)
    
    # TODO - Fit the classifier on the training set - 5 points
    svc_rbf.fit(X_train,Y_train)
    
    # TODO - Find the prediction and accuracy on the training set - 5 points
    Yhat_svc_rbf_train = svc_rbf.predict(X_train)
    acc_train = svc_rbf.score(X_train, Y_train)
    
    # Adding testing accuracy to acc_train_svm
    acc_train_svm_rbf.append(acc_train)
    # print('Train Accuracy = {0:f}'.format(acc_train))
    
    # TODO - Find the prediction and accuracy on the test set - 5 points
    Yhat_svc_rbf_test = svc_rbf.predict(X_test)
    acc_test = svc_rbf.score(X_test, Y_test)
    
    # Adding testing accuracy to acc_test_svm
    acc_test_svm_rbf.append(acc_test)
    # print('Test Accuracy = {0:f}'.format(acc_test))
    
    # Appending value of c for graphing purposes
    c_svm_rbf.append(c)
import warnings
warnings.filterwarnings("ignore")
acc_train_svm_rbf=[]
acc_test_svm_rbf=[]
c_svm_rbf=[]
c_values=[]
for i in range(6):
    for j in range(1,10):
        c_values.append((10**i)*j)
for c in tqdm(c_values):
    svm_rbf(c/10000)
plt.plot(c_svm_rbf,acc_train_svm_rbf,'.-',color='green')
plt.plot(c_svm_rbf,acc_test_svm_rbf,'.-',color='red')


# Use the following function to have a legend
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
plt.grid()
plt.savefig('svm_rbf.png', bbox_inches='tight')

from sklearn import svm
acc_train_svm_poly = []
acc_test_svm_poly = []
c_svm_poly = []
def svm_polynomial(c):
    # TODO - Create an object of svm.SVC(probability = False, kernel = 'poly', C = c) - 5 points
    svc_polynomial = svm.SVC(probability = False, kernel = 'poly', C = c)
    
    A = X_train  # First 300 rows of training set.
    B = Y_train 
    C = X_test   # First 100 rows of test set.
    D = Y_test
    
    # TODO - Fit the classifier on the training set - 5 points
    # Use A and B to train and C and D to test.
    svc_polynomial.fit(A,B)
    
    # TODO - Find the prediction and accuracy on the training set - 5 points
    Yhat_svc_poly_train = svc_polynomial.predict(A)
    acc_train = svc_polynomial.score(A, B)
    
    # Adding testing accuracy to acc_train_svm
    acc_train_svm_poly.append(acc_train)
    #print('Train Accuracy = {0:f}'.format(acc_train))
    
    # TODO - Find the prediction and accuracy on the test set - 5 points
    Yhat_svc_poly_test = svc_polynomial.predict(C)
    acc_test = svc_polynomial.score(C, D)
    
    # Adding testing accuracy to acc_test_svm
    acc_test_svm_poly.append(acc_test)
    #print('Test Accuracy = {0:f}'.format(acc_test))
    
    # Appending value of c for graphing purposes
    c_svm_poly.append(c)

import warnings
warnings.filterwarnings("ignore")
acc_train_svm_poly=[]
acc_test_svm_poly=[]
c_svm_poly=[]
c_values=[]
for i in range(6):
    for j in range(1,10):
        c_values.append((10**i)*j)
for c in tqdm(c_values):
    svm_polynomial(c/10000)
plt.plot(c_svm_poly,acc_train_svm_poly,'.-',color='green')
plt.plot(c_svm_poly,acc_test_svm_poly,'.-',color='red')


# Use the following function to have a legend
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
plt.grid()
plt.savefig('svm_poly.png', bbox_inches='tight')

'''