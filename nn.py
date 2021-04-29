import os
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # It is important in neural networks to scale the date
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
import sys
np.set_printoptions(threshold=sys.maxsize)
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

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 3))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


def f(z):
    return 1 / (1 + np.exp(-z))

def f_deriv(z):
    return f(z) * (1 - f(z))

def relu(z):
    return np.maximum(0,z)

def relu_deriv(z):
    return np.where(z<0,0,1)

def lk_relu(z):
    y1 = ((z > 0) * z)                                                 
    y2 = ((z <= 0) * z * 0.01)                                         
    return y1 + y2 

def lk_relu_deriv(z):
    return np.where(z<0,0.01,1)
# tanh activation function
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def tanh_deriv(z):
    return 1-f(z)*f(z)

def setup_and_init_weights(nn_structure):
    W = {} #creating a dictionary i.e. a set of key: value pairs
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = np.random.random_sample((nn_structure[l], nn_structure[l-1])) #Return “continuous uniform” random floats in the half-open interval [0.0, 1.0). 
        b[l] = np.random.random_sample((nn_structure[l],))
    return W, b

def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

def feed_forward(x, W, b):
    a = {1: x} # create a dictionary for holding the a values for all levels
    z = { } # create a dictionary for holding the z values for all the layers
    for l in range(1, len(W) + 1): # for each layer
        node_in = a[l]
        z[l+1] = W[l].dot(node_in) + b[l]  # z^(l+1) = W^(l)*a^(l) + b^(l)
        # a[l+1] = f(z[l+1]) # a^(l+1) = f(z^(l+1))
        # using ReLU activation function
        # a[l+1] = relu(z[l+1])
        # using tanh activation function
        # a[l+1] = tanh(z[l+1])
        # using leaky ReLU activation function
        a[l+1] = lk_relu(z[l+1])
    return a, z

def calculate_out_layer_delta(y, a_out, z_out):
    # delta^(nl) = -(y_i - a_i^(nl)) * f'(z_i^(nl))
    # return -(y-a_out) * f_deriv(z_out)
    # using ReLU activation function
    # return -(y-a_out) * relu_deriv(z_out)
    # using tanh activation function
    # return -(y-a_out) * tanh_deriv(z_out)
    # using leaky ReLU activation function
    return -(y-a_out) * lk_relu_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    # return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)
    # using ReLU activation function
    # return np.dot(np.transpose(w_l), delta_plus_1) * relu_deriv(z_l)
    # using tanh activation function
    # return np.dot(np.transpose(w_l), delta_plus_1) * tanh_deriv(z_l)
    # using leaky ReLU activation function
    return np.dot(np.transpose(w_l), delta_plus_1) * lk_relu_deriv(z_l)

def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    lbd = 0.001
    N = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(N):
            delta = {}
            # perform the feed forward pass and return the stored a and z values, to be used in the
            # gradient descent step
            a, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], a[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-a[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(a^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(a[l][:,np.newaxis]))# np.newaxis increase the number of dimensions
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/N * tri_W[l])
            b[l] += -alpha * (1.0/N * tri_b[l])
        # Add a regularization term to the cost function
        
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/N * tri_W[l]+lbd*W[l])
            b[l] += -alpha * (1.0/N * tri_b[l])
        
        # complete the average cost calculation
        avg_cost = 1.0/N * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


def predict_y(W, b, X, n_layers):
    N = X.shape[0]
    y = np.zeros((N,))
    for i in range(N):
        a, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(a[n_layers])
    return y

fo=open('y_pred.txt','w')
'''
allFiles1=getListOfFiles('/data_1/un/mlproject/data/Charlock')
allFiles2=getListOfFiles('/data_1/un/mlproject/data/Common Chickweed')
allFiles3=getListOfFiles('/data_1/un/mlproject/data/Shepherd’s Purse')
allFiles=allFiles1+allFiles2+allFiles3
allFiles.sort()
min_w=49
min_h=49

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)
print(y_train[0:4])
print(y_v_train[0:4])


np.save( 'X_train.npy' , X_train )
np.save( 'y_v_train.npy' , y_v_train )
np.save( 'X_test.npy', X_test)
np.save( 'y_test.npy', y_test)
'''
X_train=np.load('X_train.npy')
y_v_train=np.load('y_v_train.npy')
X_test=np.load('X_test.npy')
y_test=np.load('y_test.npy')

print(X_train.shape)
print(y_v_train.shape)
print(X_test.shape)
nn_structure = [2401, 1200, 3]
# using 50 hidden layers
# nn_structure = [64, 50, 10]
# train the NN
W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train, 100)
# using 5000 iterations
# W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train, 5000)
print('end')

pickle.dump( W , open( 'weights.pkl' , 'wb' ) )
pickle.dump( b , open( 'bias.pkl' , 'wb' ) )
pickle.dump( avg_cost_func , open( 'acf.pkl' , 'wb' ) )

y_pred = predict_y(W, b, X_test, 3)
y_pred1 =predict_y(W, b, X_train, 3)
fo.write(str(y_pred1)+'\n')
fo.write(str(y_v_train)+'\n')
fo.write(str(y_pred)+'\n')
fo.write(str(y_test)+'\n')
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))

# plot the avg_cost_func
plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.savefig('avg_cost_nnlr.png', bbox_inches='tight')