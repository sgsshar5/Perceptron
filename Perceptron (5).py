#!/usr/bin/env python
# coding: utf-8

# In[106]:


import numpy as np

np.random.seed(15)
np.seterr(all='ignore')  # to ignore overflow warning



class Perceptron(object):
    """
    This class contains utilities for training and testing the perceptron model.
1. The perceptron is trained on the given dataset using train(self,data,label).
2. activation score(self,X) returns the activation score value by combining the weight and feature vector X into a dot product.
3. For a feature vector in X, predict(self,X) provides the projected value of the datapoint, which is either 1 or -1.
4. The perceptron model with L2 regularisation is trained and tested using trainWithRegularisation(self, data, label, lamb).
5.activation score
By doing a dot product of weight and feature vector X, WithRegularisation(self, X) delivers the activation score value.
6.predictWithRegularisation(self, X) provides the projected value of a datapoint for a feature vector in X, which is either 1 or -1.
    """

    def __init__(s, r, ep):
        """
        Initialiser function
        :param rate:
        :param epoch:
        """
        s.r = r
        s.ep = ep
        s.lbl = ""
        s.lmb = 0
    
    
    def trn(s, sample, lbl):
        """
        Training function without regularisation
        :param data:
        :param label:
        :return:
        """
        s.lbl = lbl
        sample[:, 4] = np.where(sample[:, 4] == s.lbl, 1, -1)
        np.random.shuffle(sample)
        X = sample[:, 0:4].astype(float)
        y = sample[:, 4].astype(float)
        s.wt = np.zeros(X.shape[1])  # Initialising weight vector to be zero
        s.b = 0  # Initialising bias to be zero
        for i in range(s.ep):
            for i, target in zip(X, y):
                d = s.r * (target - s.pred(i))  # calling predict function
                s.wt = s.wt + (d * i)  # updating weights
                s.b = s.b + d  # updating bias
        return s

    

    def trnWreg(s, sample, lbl, lmb):
        """
        Training function with regularisation
        :param data:
        :param label:
        :param lamb:
        :return:
        """
        s.lbl = lbl
        s.lmb = lmb
        sample[:, 4] = np.where(sample[:, 4] == s.lbl, 1, -1)
        np.random.shuffle(sample)
        X = sample[:, 0:4].astype(np.float64)
        y = sample[:, 4].astype(np.float64)
        s.wt = np.zeros(X.shape[1], dtype=np.float128)  # Initialising weight vector to be zero
        s.b = 0  # Initialsing bias to be zero
        for i in range(s.ep):
            for i, target in zip(X, y):
                d = s.r * (target - s.predwreg(i))  # calling predict function
                s.wt = (1 - (2 * s.lmb)) * s.wt + (d * i)  # updating weights
                s.b = s.b + d  # updating bias
        return s
    def a_score(s, X):
        """
        Activation function
        :param X:
        :return:
        """
        X = X.astype(float)
        return np.dot(X, s.wt) + s.b  # calculating activation score value
    def a_wrg(s, X):
        """
        Activation function
        :param X:
        :return:
        """
        X = X.astype(np.float128)
        return np.dot(X, s.wt) + s.b   # calculating activation score value
    
    def pred(s, X):
        """
        Prediction function
        :param X:
        :return:
        """
        return np.where(s.a_score(X) >= 0.0, 1, -1)  # Predicting based on threshold


    def predwreg(s, X):
        """
        Prediction function
        :param X:
        :return:
        """
        return np.where(s.a_wrg(X) >= 0.0, 1, -1)  # Predicting based on threshold



def acc(actual, predicted):
    """
    Accuracy function
    :param actual:
    :param predicted:
    :return:
    """
    correct = 0
    for i in range(len(actual)):
        if int(actual[i]) == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def read(filename):
    """
    File loader and reader function
    :param filename:
    :return:
    """
    arr = []
    with open(filename) as file:
        for line in file:
            arr.append(line.rstrip().split(","))
        arr = np.array(arr)
        l1 = arr[arr[:, 4] == "class-1"]
        l2 = arr[arr[:, 4] == "class-2"]
        l3 = arr[arr[:, 4] == "class-3"]
    return l1, l2, l3


"""
Reading test and training files
"""
C1tr, C2tr, C3tr = read("/Users/shivamsharma/Downloads/CA1data 2/train.data")
C1ts, C2ts, C3ts = read("/Users/shivamsharma/Downloads/CA1data 2/test.data")

"""
Q3. Use the binary perceptron to train classifiers to discriminate between the following pairs:
"""
print("----------------------------------------------------------------------")
print("Q3. Binary Perceptron Accuracy\n----------------------------------------------------------------------")
print("Accuracy                     Classes                                   ")
print("          class 1 vs class 2  class 2 vs class 3  class 1 vs class 3 ")

# a) Class 1 vs Class 2
c1vc2 = Perceptron(0.01, 20)
c1vc2t = np.concatenate((C1tr, C2tr), axis=0)
c1vc2.trn(c1vc2t, "class-1")
# Calculating Training Accuracy
accuracy1v2t = []
c1vc2t_predict = c1vc2.pred(c1vc2t[:, 0:4])
accuracy1v2t.append(acc(c1vc2t[:, 4], c1vc2t_predict))

# Calculating Testing Accuracy
accuracy1v2ts = []
c1vc2ts = np.concatenate((C1ts, C2ts), axis=0)
c1vc2ts[:, 4] = np.where(c1vc2ts[:, 4] == "class-1", 1, -1)
c1vc2t_predict = c1vc2.pred(c1vc2ts[:, 0:4])
accuracy1v2ts.append(acc(c1vc2ts[:, 4], c1vc2t_predict))

# b) Class 2 vs Class 3
c2vc3 = Perceptron(0.01, 20)
c2vc3t = np.concatenate((C2tr, C3tr), axis=0)
c2vc3.trn(c2vc3t, "class-2")

# Calculating Training Accuracy
accuracy2v3t = []
c2vc3t_predict = c2vc3.pred(c2vc3t[:, 0:4])
accuracy2v3t.append(acc(c2vc3t[:, 4], c2vc3t_predict))

# Calculating Testing Accuracy
accuracy2v3ts = []
c2vc3ts = np.concatenate((C2ts, C3ts), axis=0)
c2vc3ts[:, 4] = np.where(c2vc3ts[:, 4] == "class-2", 1, -1)
c2vc3ts_predict = c2vc3.pred(c2vc3ts[:, 0:4])
accuracy2v3ts.append(acc(c2vc3ts[:, 4], c2vc3ts_predict))

# c) Class 1 vs Class 3
c1vc3 = Perceptron(0.01, 20)
c1vc3t = np.concatenate((C1tr, C3tr), axis=0)
c1vc3.trn(c1vc3t, "class-1")
# Calculating Training Accuracy
accuracy1v3t = []
c1vc3t_predict = c1vc3.pred(c1vc3t[:, 0:4])
accuracy1v3t.append(acc(class1v3Training[:, 4], c1vc3t_predict))

# Calculating Testing Accuracy
accuracy1v3ts = []
c1vc3ts = np.concatenate((C1ts, C3ts), axis=0)
c1vc3ts[:, 4] = np.where(c1vc3ts[:, 4] == "class-1", 1, -1)
c1vc3ts_predict = c1vc3.pred(c1vc3ts[:, 0:4])
accuracy1v3t.append(acc(c1vc3ts[:, 4], c1vc3ts_predict))

print("Training      ", sum(accuracy1v2t) / len(accuracy1v2t), "              ",
      sum(accuracy2v3t) / len(accuracy2v3t), "             ",
      sum(accuracy1v3t) / len(accuracy1v3t), "        ")

print("Testing      ", sum(accuracy1v2ts) / len(accuracy1v2ts), "              ",
      sum(accuracy2v3ts) / len(accuracy2v3ts), "             ",
      sum(accuracy1v3t) / len(accuracy1v3t), "        ")

print("------------------------------------------------------------------------")

"""
Q4. Extend the binary perceptron that you implemented in part (2) above to perform multi-class classification using the 
1-vs-rest approach. 
"""
print("Q4. One vs Rest Approach Multi-Class Classification \n----------------------------------------------------------"
      "--------------")
print("Accuracy           Classes            ")
print("          class 1   class 2  class 3 ")

# Class 1 vs (Class 2, Class 3)
c1vrest = Perceptron(0.01, 20)
c1vrestT = np.concatenate((C1tr, C2tr), axis=0)
c1vrestT = np.concatenate((c1vrestT, C3tr), axis=0)
c1vrest.trn(c1vrestT, "class-1")

# Class 2 vs (Class 1, Class 3)
c2vrest = Perceptron(0.01, 20)
c2vrestT = np.concatenate((C2tr, C1tr), axis=0)
c2vrestT = np.concatenate((c2vrestT, C2tr), axis=0)
c2vrest.trn(c1vrestT, "class-2")

# Class 3 vs (Class 1, Class 2)
c3vrest = Perceptron(0.01, 20)
c3vrestT = np.concatenate((C3tr, C1tr), axis=0)
c3vrestT = np.concatenate((c3vrestT, C2tr), axis=0)
c3vrest.trn(c3vrestT, "class-3")

# Calculating Testing accuracy
C1ts[:, 4] = np.where(C1ts[:, 4] == "class-1", 1, -1)
C2ts[:, 4] = np.where(C2ts[:, 4] == "class-2", 2, -1)
C3ts[:, 4] = np.where(C3ts[:, 4] == "class-3", 3, -1)
c1vc2vc3ts = np.concatenate((C1ts, C2ts,), axis=0)
c1vc2vc3ts = np.concatenate((c1vc2vc3ts, C3ts,), axis=0)
TP = 0
caccuracyts = np.zeros(3)
for point in c1vc2vc3ts:
    score = []
    score.append(c1vrest.a_score(point[:4]))
    score.append(c2vrest.a_score(point[:4]))
    score.append(c3vrest.a_score(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracyts[int(point[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
C1tr[:, 4] = np.where(C1tr[:, 4] == "class-1", 1, -1)
C2tr[:, 4] = np.where(C2tr[:, 4] == "class-2", 2, -1)
C3tr[:, 4] = np.where(C3tr[:, 4] == "class-3", 3, -1)
c1vc2vc3tr = np.concatenate((C1tr, C2tr,), axis=0)
c1vc2vc3tr = np.concatenate((c1vc2vc3tr, C3tr,), axis=0)
TP = 0
caccuracyt = np.zeros(3)
for point in c1vc2vc3tr:
    score = []
    score.append(c1vrest.a_score(point[:4]))
    score.append(pc2.a_score(point[:4]))
    score.append(c3vrest.a_score(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracyt[int(point[4]) - 1] += 1
        TP += 1
trn1 = ((caccuracyt / (c1vc2vc3tr.shape[0] / 3))[0]) * 100
trn2 = ((caccuracyt / (c1vc2vc3tr.shape[0] / 3))[1]) * 100
trn3 = ((caccuracyt / (c1vc2vc3tr.shape[0] / 3))[2]) * 100
print("Training  ", tr1, "   ", tr2, "    ", tr3, "  ")
ts1 = ((caccuracyts / (c1vc2vc3ts.shape[0] / 3))[0]) * 100
ts2 = ((caccuracyts / (c1vc2vc3ts.shape[0] / 3))[1]) * 100
ts3 = ((caccuracyts / (c1vc2vc3ts.shape[0] / 3))[2]) * 100
print("Testing   ", tes1, "   ", tes2, "    ", tes3, "  ")
print("------------------------------------------------------------------------")

"""
Q5. Add an l2 regularisation term to your multi-class classifier implemented in question. Set the regularisation
 coefficient to 0.01, 0.1, 1.0, 10.0, 100.0 and compare the train and test classification accuracy for each of the
 three classes.
"""
print("Q5. Multi- Classification with L2 regularisation \n----------------------------------------------------------"
      "--------------")
print("Accuracy  Lambda               Classes             ")
print("                     class 1   class 2   class 3 ")

"""Lambda 0.01"""
# Class 1 vs (Class 2, Class 3)
c1vc2c3 = Perceptron(0.01, 20)
c1vc2c3.trnWreg(c1vrestT, "class-1", 0.01)
# # Class 2 vs (Class 1, Class 3)
c2vc1c3 = Perceptron(0.01, 20)
c2vc1c3.trnWreg(c1vrestT, "class-2", 0.01)
# Class 3 vs (Class 1, Class 2)
c3vc1c2 = Perceptron(0.01, 20)
c3vc1c2.trnWreg(c3vrestT, "class-3", 0.01)
# Calculating Testing accuracy
TP = 0
caccuracylmbTs = np.zeros(3)
for point in c1vc2vc3ts:
    score = []
    score.append(c1vc2c3.a_wrg(point[:4]))
    score.append(c2vc1c3.a_wrg(point[:4]))
    score.append(c3vc1c2.a_wrg(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracylmbTs[int(point[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
caccuracylmbTr = np.zeros(3)
for point in c1vc2vc3tr:
    score = []
    score.append(c1vc2c3.a_wrg(point[:4]))
    score.append(c2vc1c3.a_wrg(point[:4]))
    score.append(c3vc1c2.a_wrg(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracylmbTr[int(point[4]) - 1] += 1
        TP += 1

trnl1 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[0]) * 100
trnl2 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[1]) * 100
trnl3 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[2]) * 100
print("Training    0.01     ", trl1, "   ", trl2, "    ", trl3, "    ")
tsl1 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[0]) * 100
tsl2 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[1]) * 100
tsl3 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[2]) * 100
print("Testing     0.01     ", tsl1, "   ", tsl2, "    ", tsl3, "    \n")

"""Lambda 0.1"""
# Class 1 vs (Class 2, Class 3)
c1vc2c3l = Perceptron(0.01, 20)
c1vc2c3l.trnWreg(c1vrestT, "class-1", 0.1)

# Class 2 vs (Class 1, Class 3)
c2vc1c3l = Perceptron(0.01, 20)
c2vc1c3l.trnWreg(c1vrestT, "class-2", 0.1)

# Class 3 vs (Class 1, Class 2)
c3vc1c2l = Perceptron(0.01, 20)
c3vc1c2l.trnWreg(c3vrestT, "class-3", 0.1)

# Calculating Testing accuracy
TP = 0
caccuracylmbTs = np.zeros(3)
for point in c1vc2vc3ts:
    score = []
    score.append(c1vc2c3l.a_wrg(point[:4]))
    score.append(c2vc1c3l.a_wrg(point[:4]))
    score.append(c3vc1c2l.a_wrg(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracylmbTs[int(point[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
caccuracylmbTr = np.zeros(3)

for point in c1vc2vc3tr:
    score = []
    score.append(c1vc2c3l.a_wrg(point[:4]))
    score.append(c2vc1c3l.a_wrg(point[:4]))
    score.append(c3vc1c2l.a_wrg(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracylmbTr[int(point[4]) - 1] += 1
        TP += 1

trnl4 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[0]) * 100
trnl5 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[1]) * 100
trnl6 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[2]) * 100
print("Training    0.1      ", trl4, "  ", trl5, "     ", trl6, "    ")
tsl4 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[0]) * 100
tsl5 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[1]) * 100
tsl6 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[2]) * 100
print("Testing     0.1      ", tsl4, "  ", tsl5, "     ", tsl6, "    \n")

"""Lambda 1"""
# Class 1 vs (Class 2, Class 3)
c1vc2c3lm = Perceptron(0.01, 20)
c1vc2c3lm.trnWreg(c1vrestT, "class-1", 1)

# Class 2 vs (Class 1, Class 3)
c2vc1c3lm = Perceptron(0.01, 20)
c2vc1c3lm.trnWreg(c1vrestT, "class-2", 1)

# Class 3 vs (Class 1, Class 2)
c3vc1c2lm = Perceptron(0.01, 20)
c3vc1c2lm.trnWreg(c3vrestT, "class-3", 1)
# Calculating Testing accuracy
TP = 0
caccuracylmbTs = np.zeros(3)
for point in c1vc2vc3ts:
    score = []
    score.append(c1vc2c3lm.a_wrg(point[:4]))
    score.append(c2vc1c3lm.a_wrg(point[:4]))
    score.append(c3vc1c2lm.a_wrg(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracylmbTs[int(point[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
caccuracylmbTr = np.zeros(3)
for point in c1vc2vc3tr:
    score = []
    score.append(c1vc2c3lm.a_wrg(point[:4]))
    score.append(c2vc1c3lm.a_wrg(point[:4]))
    score.append(c3vc1c2lm.a_wrg(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracylmbTr[int(point[4]) - 1] += 1
        TP += 1

trnl7 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[0]) * 100
trnl8 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[1]) * 100
trnl9 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[2]) * 100
print("Training      1       ", trl7, "  ", trl8, "     ", trl9, "    ")
tsl7 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[0]) * 100
tsl8 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[1]) * 100
tsl9 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[2]) * 100
print("Testing       1       ", tsl7, " ",  tsl8, "     ",  tsl9, "    \n")

"""Lambda 10"""
# Class 1 vs (Class 2, Class 3)
c1vc2c3lm10 = Perceptron(0.01, 20)
c1vc2c3lm10.trnWreg(c1vrestT, "class-1", 10)

# Class 2 vs (Class 1, Class 3)
c2vc1c3lm10 = Perceptron(0.01, 20)
c2vc1c3lm10.trnWreg(c1vrestT, "class-2", 10)

# Class 3 vs (Class 1, Class 2)
c3vc1c2lm10 = Perceptron(0.01, 20)
c3vc1c2lm10.trnWreg(c3vrestT, "class-3", 10)

# Calculating Testing accuracy
TP = 0
caccuracylmbTs = np.zeros(3)
for point in c1vc2vc3ts:
    score = []
    score.append(c1vc2c3lm10.a_wrg(point[:4]))
    score.append(c2vc1c3lm10.a_wrg(point[:4]))
    score.append(c3vc1c2lm10.a_wrg(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracylmbTs[int(point[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
caccuracylmbTr = np.zeros(3)
for point in c1vc2vc3tr:
    score = []
    score.append(c1vc2c3lm10.a_wrg(point[:4]))
    score.append(c2vc1c3lm10.a_wrg(point[:4]))
    score.append(c3vc1c2lm10.a_wrg(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracylmbTr[int(point[4]) - 1] += 1
        TP += 1

trnl10 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[0]) * 100
trnl11 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[1]) * 100
trnl12 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[2]) * 100
print("Training      10     ", trl10, "   ", trl11, "   ", trl12, "    ")
tsl10 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[0]) * 100
tsl11 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[1]) * 100
tsl12 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[2]) * 100
print("Testing       10      ", tsl10, "   ", tsl11, "   ", tsl12, "    \n")

"""Lambda 100"""
# Class 1 vs (Class 2, Class 3)
c1vc2c3lm100 = Perceptron(0.01, 20)
c1vc2c3lm100.trnWreg(c1vrestT, "class-1", 100)

# Class 2 vs (Class 1, Class 3)
c2vc1c3lm100 = Perceptron(0.01, 20)
c2vc1c3lm100.trnWreg(c1vrestT, "class-2", 100)

# Class 3 vs (Class 1, Class 2)
c3vc1c2lm100 = Perceptron(0.01, 20)
c3vc1c2lm100.trnWreg(c3vrestT, "class-3", 100)

# Calculating Testing accuracy
TP = 0
caccuracylmbTs = np.zeros(3)
for point in c1vc2vc3ts:
    score = []
    score.append(c1vc2c3lm100.a_wrg(point[:4]))
    score.append(c2vc1c3lm100.a_wrg(point[:4]))
    score.append(c3vc1c2lm100.a_wrg(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracylmbTs[int(point[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
caccuracylmbTr = np.zeros(3)
for point in c1vc2vc3tr:
    score = []
    score.append(c1vc2c3lm100.a_wrg(point[:4]))
    score.append(c2vc1c3lm100.a_wrg(point[:4]))
    score.append(c3vc1c2lm100.a_wrg(point[:4]))
    prediction = np.argmax(score) + 1
    if prediction == int(point[4]):
        caccuracylmbTr[int(point[4]) - 1] += 1
        TP += 1

trnl13 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[0]) * 100
trnl14 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[1]) * 100
trnl15 = ((caccuracylmbTr / (c1vc2vc3tr.shape[0] / 3))[2]) * 100
print("Training      100     ", trl10, "   ", trl11, "   ", trl12, "   ")
tsl13 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[0]) * 100
tsl14 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[1]) * 100
tsl15 = ((caccuracylmbTs / (c1vc2vc3ts.shape[0] / 3))[2]) * 100
print("Testing       100     ", tsl10, "   ", tsl11, "   ", tsl12, "    \n")


# In[ ]:





# In[ ]:





# In[ ]:




