import numpy as np
import pandas as pd
import math

def import_train_data(filename):
    td = open(filename,"r")
    rawData = []
    for line in td:
        row = line.rstrip('\r\n').split(',')
        for i in range(1, len(row)):
            row[i] = float(row[i])
        mail = row[1:-1]
        label = row[-1]
        rawData.append((np.array(mail),label))
    return rawData

def sigmoid(z):
    if z > 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        z = math.exp(z)
    return z / (1.0 + z)

def f(w, b, x):
    p = sigmoid((w * x).sum() + b)
    if p >= 0.5:
        return 0
    if p < 0.5:
        return 1

def f1(w, b, x):
    return sigmoid((w * x).sum() + b)


def max_likelihood(data):
    b = np.mat(np.zeros(57))
    N = len(data)
    count = 0.0
    miu0 = np.mat(np.zeros(57))
    miu1 = np.mat(np.zeros(57))
    thegma0 = np.mat(np.zeros([57, 57]))
    thegma1 = np.mat(np.zeros([57, 57]))
    for line in data:
        if line[1] == 0:
            miu0 += np.mat(line[0])
            count += 1
        else:
            miu1 += np.mat(line[0])
    miu0 = (miu0/count).T
    miu1 = (miu1/(N-count)).T
    for line in data:
        x = np.mat(line[0]).T
        if line[1] == 0:
            thegma0 += (x - miu0) * (x - miu0).T
        else:
            thegma1 += (x - miu1) * (x - miu1).T
    thegma0 = thegma0/count
    thegma1 = thegma1/(N-count)
    thegma = thegma0 * (count/N) + thegma1 * (1-count/N)
    w = (miu0 - miu1).T * thegma**-1
    b = math.log(count/(N-count)) + miu1.T * thegma**-1 * miu1 * 0.5 - miu0.T * thegma**-1 * miu0 * 0.5
    #print w.shape,b.shape,thegma.shape,miu0.shape,miu1.shape,miu0.T
    return np.array(w),b[0,0]

def cross_entropy(w,b):
    loss = 0
    for eachmail in trainset:
        x = eachmail[0]
        fx = f1(w,b,x)
        label = eachmail[1]
        loss += -(label*math.log(fx)+(1-label)*math.log(fx))
    return loss

def gradient(w, b, x, y):
    y_ = f1(w, b, x) - y
    return y_ * x, y_

def adagrad(iteration_num,l_rate):
    w0 = (np.random.random(57) - 0.5)/1000
    b0 = (np.random.random() -0.5)/1000
    gw_acc = np.zeros(57)
    gb_acc = 0.0
    for i in range(iteration_num):
        print cross_entropy(w0,b0)
        gw_ada = np.zeros(57)
        gb_ada = 0.0
        for mail in trainset:
            x,y = mail[0],mail[1]
            gw,gb = gradient(w0,b0,x,y)
            gw_ada += gw
            gb_ada += gb
        gw_acc += gw_ada ** 2
        gb_acc += gb_ada ** 2
        w0 -= l_rate*gw_ada/np.sqrt(gw_acc)
        b0 -= l_rate*gb_ada/np.sqrt(gb_acc)
    return w0,b0

def accuracy(w,b):
    rightsum = 0.0
    count = 0
    for eachmail in trainset:
        count += 1
        x = eachmail[0]
        label = eachmail[1]
        if f(w, b, x) == label:
            rightsum += 1
    return rightsum / count


trainset = import_train_data("spam_data\spam_train.csv")
w,b = max_likelihood(trainset)
w0,b0 = adagrad(1000,0.01)


print cross_entropy(w,b)
print accuracy(w,b)
print cross_entropy(w0,b0)
print 1 - accuracy(w0,b0)
