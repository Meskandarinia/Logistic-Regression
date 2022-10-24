

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from sklearn.naive_bayes import BernoulliNB, GaussianNB



data= pd.read_csv(r"C:\Users\Mohammad\Downloads\train.csv")
data=data.drop(["Genus" , "Species", "RecordID","MFCCs_ 1"], axis =1)
x_data=data.drop(["Family"],axis=1)
y_data=data["Family"]
y_data[y_data == "Leptodactylidae"] = 0
y_data[y_data == "Hylidae"] = 1
y_data[y_data == "Dendrobatidae"] = 2
y_data[y_data == "Bufonidae"] = 3
x_data=x_data.to_numpy()
y_data=y_data.to_numpy()
x_data=np.array(x_data, dtype=float)
y_data=np.array(y_data,dtype=float)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2,random_state=3)
x_train_without_val, x_val, y_train_without_val, y_val = train_test_split(x_train, y_train, test_size=0.2)


# #A1

def sigmoid(X, W):
    z = np.dot(X, W)
    return 1 / (1 + np.exp(-z))

def gradient_ascent(X, h, y, regularization, C, weight):
      if regularization == True:
        gradient = np.dot(X.T, y - h) + C * weight
      else:
        gradient = np.dot(X.T, y - h)
      return gradient

def logistic_regression(num_iter, x_train, x_test, y_train, y_test, learning_rate, regularization_condition ,c):
  acc_train = []
  acc_test = []
  start_time = time.time()
  intercept = np.ones((x_train.shape[0], 1))
  x_train = np.concatenate((intercept, x_train), axis=1)
  intercept_test = np.ones((x_test.shape[0], 1))
  x_test = np.concatenate((intercept_test, x_test), axis=1)
  alpha = np.zeros(x_train.shape[1])
  for i in range(num_iter):
      h = sigmoid(x_train, alpha)
      gradient = gradient_ascent(x_train, h, y_train, regularization_condition, c, alpha) 
      alpha = alpha + learning_rate * gradient
      result_train = sigmoid(x_train, alpha).round()
      result_test = sigmoid(x_test, alpha).round()
      acc_train.append(100 * accuracy_score(y_train, result_train))
      acc_test.append(100 * accuracy_score(y_test, result_test))
  clock = time.time() - start_time
  return  acc_train, acc_test, clock

learning_rate_list = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
num_iter = 100
fig = plt.figure(figsize=(12, 5))
for learning_rate in learning_rate_list:
  acc_train, acc_val, clock = logistic_regression(num_iter, x_train_without_val, x_val, y_train_without_val, y_val, learning_rate, False, 0)
  print('Learning Rate :', str(learning_rate))
  print('Number of Iteration :', str(num_iter))
  print('Training Time :', str(round(clock, 2)), 'Seconds')
  print('---------------------')
  plt.plot(acc_val)
  plt.legend(['Learning Rate: 0.1', 'Learning Rate: 0.01', 'Learning Rate: 0.005', 'Learning Rate: 0.001', 'Learning Rate: 0.0005', 'Learning Rate: 0.0001'], loc='lower right')
  plt.ylabel('Accuracy (%)')
  plt.xlabel('Iteration')
  
plt.grid()  
plt.show()



learning_rate = 0.001
num_iter = 50
acc_train, acc_test, clock = logistic_regression(num_iter, x_train, x_test, y_train, y_test, learning_rate, False, 0)
print('---------------------')
print('Learning Rate :', str(learning_rate))
print('Number of Iteration :', str(num_iter))
print('Training Time :', str(round(clock, 2)), 'Seconds')
print('Train Accuracy :', str(round(acc_train[-1], 2)), '%')
print('Test Accuracy :', str(round(acc_test[-1], 2)), '%')
print('---------------------')
fig = plt.figure(figsize=(12, 5))
plt.plot(acc_train)
plt.plot(acc_test)
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration')
plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
plt.grid()
plt.show()

# #A2
start_time = time.time()
gnb = GaussianNB()
gnb.fit(x_train, y_train)
clock = time.time() - start_time
y_train_pred = gnb.predict(x_train)
acc_train = 100 * accuracy_score(y_train, y_train_pred)
y_test_pred = gnb.predict(x_test)
acc_test = 100 * accuracy_score(y_test, y_test_pred)

print('Training Time :', str(round(clock, 2)), 'Seconds')
print('Train Accuracy is:', str(round(acc_train, 2)), '%')
print('Test Accuracy is:',str(round(acc_test, 2)), '%')

plt.plot(acc_train,'r')
plt.plot(acc_test,'c')
plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration')
plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
plt.grid()
plt.title('Model Accuracy')
plt.show()


 #A3

learning_rate = 0.01
num_iter = 100
c_list=[10, 5, 1, 0.5, 0]

fig = plt.figure(figsize=(12, 5))
for c in c_list:
  acc_train, acc_test, clock = logistic_regression(num_iter, x_train_without_val, x_val, y_train_without_val, y_val, learning_rate, True, c)
  print('Learning Rate :', str(learning_rate))
  print('Number of Iteration :', str(num_iter))
  print('Regularization  Parameter :', str(c))
  print('Training Time :', str(round(clock, 2)), 'Seconds')
  print('---------------------') 
  plt.plot(acc_test, label=str(c))
  plt.legend(['Regularization Parameter: 10', 'Regularization Parameter: 5', 'Regularization Parameter: 1', 'Regularization Parameter: 0.5', 'Regularization Parameter: 0'], loc='lower right')
  plt.ylabel('Accuracy (%)')
  plt.xlabel('Iteration')
plt.grid()
plt.show()

learning_rate = 0.01
num_iter = 50
c = 5
acc_train, acc_test, clock = logistic_regression(num_iter, x_train, x_test, y_train, y_test, learning_rate, True, c)
print('---------------------')
print('Learning Rate :', str(learning_rate))
print('Number of Iteration :', str(num_iter))
print('Training Time :', str(round(clock, 2)), 'Seconds')
print('Train Accuracy :', str(round(acc_train[-1], 2)), '%')
print('Test Accuracy :', str(round(acc_test[-1], 2)), '%')
print('---------------------')
fig = plt.figure(figsize=(12, 5))
plt.plot(acc_train)
plt.plot(acc_test)
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration')
plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
plt.grid()
plt.show()


# A4

learning_rate = 0.01
num_iter = 50
c = 5
acc_train_list_reg = []
acc_test_list_reg = []
clock_list_reg = []
acc_train_list_nb = []
acc_test_list_nb = []
clock_list_nb = []


# # Logistic Regression
for count in range(50,len(x_train),50):
  idx = np.random.choice(len(x_train),size=count)
  x_train_selected = x_train[idx,:]
  y_train_selected = y_train[idx]
  acc_train, acc_test, clock = logistic_regression(num_iter, x_train_selected, x_test, y_train_selected, y_test, learning_rate, True, c)
  acc_train_list_reg.append(acc_train[-1])
  acc_test_list_reg.append(acc_test[-1])
  clock_list_reg.append(clock)

# #GNB
for count in range(50,len(x_train),50):
  idx = np.random.choice(len(x_train),size=count)
  x_train_selected = x_train[idx,:]
  y_train_selected = y_train[idx]
  gnb = GaussianNB()
  start_time = time.time()
  gnb.fit(x_train_selected, y_train_selected)
  clock = start_time - time.time()
  y_train_selected_pred = gnb.predict(x_train_selected)
  y_test_pred = gnb.predict(x_test)
  acc_train = accuracy_score(y_train_selected, y_train_selected_pred)
  acc_test = accuracy_score(y_test, y_test_pred)
  acc_train_list_nb.append(100 * acc_train)
  acc_test_list_nb.append(100 * acc_test)
  clock_list_nb.append(clock)
  
 
  
count = range(50,len(x_train),50)

print('---------------------') 
print('Logistic Regression Average Training Time :', str(round(sum(clock_list_reg) / len(clock_list_reg), 2)), 'Seconds')
print('Gaussian Naive Bayes Average Training Time :', str(round(sum(clock_list_nb) / len(clock_list_nb), 2)), 'Seconds')
print('---------------------')
fig = plt.figure(figsize=(12, 5))
plt.plot(count, acc_train_list_reg)
plt.plot(count, acc_test_list_reg)
plt.plot(count, acc_train_list_nb)
plt.plot(count, acc_test_list_nb)
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Number of Sample in Training Set')
plt.legend(['Logistic Regression Train Accuracy', 'Logistic Regression Test Accuracy', 'Gaussian Naive Bayse Train Accuracy', 'Gaussian Naive Bayse Test Accuracy'], loc='best')
plt.grid()
plt.show()


#B1


learning_rate_list = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
num_iter = 100
acc_train_kfold_list = []
acc_test_kfold_list = []
number_of_fold=3

kf = KFold(number_of_fold)
fig = plt.figure(figsize=(12, 5))

for learning_rate in learning_rate_list:
    acc_train_kfold_list=[]
    acc_test_kfold_list=[]
    clock_list = []
    for train_index, test_index in kf.split(x_train):
        x_train_k_fold, x_val_k_fold = x_train[train_index], x_train[test_index]
        y_train_k_fold, y_val_k_fold = y_train[train_index], y_train[test_index]
        acc_train, acc_test, clock = logistic_regression(num_iter, x_train_k_fold, x_val_k_fold, y_train_k_fold, y_val_k_fold, learning_rate, False, 0)
        acc_train_kfold_list.append(acc_train)
        acc_test_kfold_list.append(acc_test)
        clock_list.append(clock)
    clock = sum(clock_list) / len(clock_list)
    acc_test_kfold_list = np.average(np.array(acc_test_kfold_list), 0)
    plt.plot(acc_test_kfold_list, label=str(learning_rate))
    print('Learning Rate :', str(learning_rate))
    print('Number of Iteration :', str(num_iter))
    print('Average Training Time :', str(round(clock, 2)), 'Seconds')
    print('---------------------')

plt.legend(['Learning Rate: 0.1', 'Learning Rate: 0.01', 'Learning Rate: 0.005', 'Learning Rate: 0.001', 'Learning Rate: 0.0005', 'Learning Rate: 0.0001'], loc='upper right')
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration')
plt.ylim([69.5, 100.5])
plt.grid()


learning_rate = 0.0005
num_iter = 50
i = 0
acc_train_kfold_list = []
acc_test_kfold_list = []
repeat_number = 3
fold_number = 3

fig = plt.figure(figsize=(12, 5))

acc_train_kfold_list = []
acc_test_kfold_list = []
clock_list = []

for count in range(repeat_number):
    kf = KFold(number_of_fold, shuffle=True)
    for train_index, test_index in kf.split(x_train):
        x_train_k_fold, x_val_k_fold = x_train[train_index], x_train[test_index]
        y_train_k_fold, y_val_k_fold = y_train[train_index], y_train[test_index]
        acc_train, acc_test, clock = logistic_regression(num_iter, x_train_k_fold, x_test, y_train_k_fold, y_test, learning_rate, False, c=0)
        acc_train_kfold_list.append(acc_train)
        acc_test_kfold_list.append(acc_test)
        clock_list.append(clock)
        i = i + 1
        print('Final Test Accuracy Number', str(i), ':', str(round(sum(acc_test) / len(acc_test), 2)), '%')
clock = sum(clock_list) / len(clock_list)
acc_train_kfold_list = np.average(np.array(acc_train_kfold_list), 0)
plt.plot(acc_train_kfold_list)
acc_test_kfold_list = np.average(np.array(acc_test_kfold_list), 0)
plt.plot(acc_test_kfold_list)

plt.legend([], loc='lower right')

plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration')
plt.grid()
print('---------------------')
print('Learning Rate :', str(learning_rate))
print('Number of Iteration :', str(num_iter))
print('Average Training Time :', str(round(clock, 2)), 'Seconds')
print('---------------------')
plt.show()

#B2



learning_rate = 0.005
num_iter = 100
c_list=[10, 5, 1, 0.5, 0]
acc_train_kfold_list = []
acc_test_kfold_list = []
number_of_fold = 3

kf = KFold(number_of_fold)
fig = plt.figure(figsize=(12, 5))

for c in c_list:
    acc_train_kfold_list=[]
    acc_test_kfold_list=[]
    clock_list = []
    for train_index, test_index in kf.split(x_train):
        x_train_k_fold, x_val_k_fold = x_train[train_index], x_train[test_index]
        y_train_k_fold, y_val_k_fold = y_train[train_index], y_train[test_index]
        acc_train, acc_test, clock = logistic_regression(num_iter, x_train_k_fold, x_val_k_fold, y_train_k_fold, y_val_k_fold, learning_rate, True, c)
        acc_train_kfold_list.append(acc_train)
        acc_test_kfold_list.append(acc_test)
        clock_list.append(clock)
    clock = sum(clock_list) / len(clock_list)
    acc_test_kfold_list = np.average(np.array(acc_test_kfold_list), 0)
    plt.plot(acc_test_kfold_list, label=str(learning_rate))
    print('Learning Rate :', str(learning_rate))
    print('Number of Iteration :', str(num_iter))
    print('Regularization  Parameter :', str(c))
    print('Average Training Time :', str(round(clock, 2)), 'Seconds')
    print('---------------------')

plt.legend(['Regularization Parameter: 10', 'Regularization Parameter: 5', 'Regularization Parameter: 1', 'Regularization Parameter: 0.5', 'Regularization Parameter: 0'], loc='lower right')
plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration')
plt.grid()



learning_rate = 0.005
num_iter = 50
c = 0.5
i = 0
acc_train_kfold_list = []
acc_test_kfold_list = []
repeat_number = 3
fold_number = 3

fig = plt.figure(figsize=(12, 5))

acc_train_kfold_list = []
acc_test_kfold_list = []
clock_list = []
print('---------------------')
for count in range(repeat_number):
    kf = KFold(number_of_fold, shuffle=True)
    for train_index, test_index in kf.split(x_train):
        x_train_k_fold, x_val_k_fold = x_train[train_index], x_train[test_index]
        y_train_k_fold, y_val_k_fold = y_train[train_index], y_train[test_index]
        acc_train, acc_test, clock = logistic_regression(num_iter, x_train_k_fold, x_test, y_train_k_fold, y_test, learning_rate, True, c)
        acc_train_kfold_list.append(acc_train)
        acc_test_kfold_list.append(acc_test)
        clock_list.append(clock)
        i = i + 1
        print('Final Test Accuracy Number', str(i), ':', str(round(sum(acc_test) / len(acc_test), 2)), '%')
clock = sum(clock_list) / len(clock_list)
acc_train_kfold_list = np.average(np.array(acc_train_kfold_list), 0)
plt.plot(acc_train_kfold_list)
acc_test_kfold_list = np.average(np.array(acc_test_kfold_list), 0)
plt.plot(acc_test_kfold_list)

plt.legend([ 'Test Accuracy'], loc='lower right')
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration')
plt.grid()
print('---------------------')
print('Learning Rate :', str(learning_rate))
print('Number of Iteration :', str(num_iter))
print('Regularization  Parameter :', str(c))
print('Average Training Time :', str(round(clock, 2)), 'Seconds')
print('---------------------')
plt.show()









learning_rate = 0.01
num_iter = 50
c = 5
acc_train_list_reg = []
acc_test_list_reg = []
clock_list_reg = []
acc_train_list_nb = []
acc_test_list_nb = []
clock_list_nb = []


# Logistic Regression

for count in range(50,len(x_train),50):
  idx = np.random.choice(len(x_train),size=count)
  x_train_selected = x_train[idx,:]
  y_train_selected = y_train[idx]
  acc_train, acc_test, clock = logistic_regression(num_iter, x_train_selected, x_test, y_train_selected, y_test, learning_rate, True, c)
  acc_train_list_reg.append(acc_train[-1])
  acc_test_list_reg.append(acc_test[-1])
  clock_list_reg.append(clock)

#Bernoulli
for count in range(50,len(x_train),50):
  idx = np.random.choice(len(x_train),size=count)
  x_train_selected = x_train[idx,:]
  y_train_selected = y_train[idx]
  gnb = BernoulliNB()
  start_time = time.time()
  gnb.fit(x_train_selected, y_train_selected)
  clock = start_time - time.time()
  y_train_selected_pred = gnb.predict(x_train_selected)
  y_test_pred = gnb.predict(x_test)
  acc_train = accuracy_score(y_train_selected, y_train_selected_pred)
  acc_test = accuracy_score(y_test, y_test_pred)
  acc_train_list_nb.append(100 * acc_train)
  acc_test_list_nb.append(100 * acc_test)
  clock_list_nb.append(clock)
  
 
  
count = range(50,len(x_train),50)

print('---------------------') 
print('Logistic Regression Average Training Time :', str(round(sum(clock_list_reg) / len(clock_list_reg), 2)), 'Seconds')
print('Bernoulli Naive Bayes Average Training Time :', str(round(sum(clock_list_nb) / len(clock_list_nb), 2)), 'Seconds')
print('---------------------')
fig = plt.figure(figsize=(12, 5))
plt.plot(count, acc_train_list_reg)
plt.plot(count, acc_test_list_reg)
plt.plot(count, acc_train_list_nb)
plt.plot(count, acc_test_list_nb)
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Number of Sample in Training Set')
plt.legend(['Logistic Regression Train Accuracy', 'Logistic Regression Test Accuracy', 'Bernoulli Naive Bayse Train Accuracy', 'Bernoulli Naive Bayse Test Accuracy'], loc='best')
plt.grid()
plt.show()


